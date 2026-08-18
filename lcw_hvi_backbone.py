import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# LCW-HVI-Restormer
# ============================================================================
# Ideia principal:
# 1. A imagem RGB entra no intervalo [0,1].
# 2. RGB e convertido para HVI, separando cor (H,V) e intensidade (I).
# 3. A intensidade recebe informacao Wavelet e um ramo Restormer proprio.
# 4. A cor recebe um ramo Restormer separado.
# 5. Os dois ramos trocam informacao por uma fusao residual controlada.
# 6. A intensidade e corrigida por uma curva limitada, iniciada em identidade.
# 7. A cor pode ficar travada ou receber apenas uma pequena correcao limitada.
# 8. HVI e convertido novamente para RGB e limitado a [0,1].
#
# Diferencas para a LCWSwinNet antiga:
# - nao existe base RGB livre;
# - nao existe residual RGB livre;
# - nao existe window partition/shifted window;
# - nao existe downscaling no caminho de reconstrucao;
# - luminosidade e cor sao tratadas separadamente.
# ============================================================================



# HVI: conversao RGB <-> HVI
# Baseada na transformacao oficial do HVI-CIDNet, com uma adaptacao de
# estabilidade: k pode ficar fixo durante os primeiros experimentos.

class HVITransform(nn.Module):
    def __init__(self, density_k=0.2, learnable_k=False):
        super().__init__()
        density_k = float(density_k)
        if density_k <= 0.0:
            raise ValueError("density_k precisa ser maior que zero.")
        if learnable_k:
            self.density_k = nn.Parameter(torch.tensor([density_k], dtype=torch.float32))
        else:
            self.register_buffer("density_k", torch.tensor([density_k], dtype=torch.float32))

    def current_k(self):
        # Evita valores nao positivos caso k seja aprendivel.
        return self.density_k.clamp_min(1e-3)

    def rgb_to_hvi(self, img):
        # A transformacao HVI foi definida para RGB em [0,1].
        img = img.clamp(0.0, 1.0)
        eps = 1e-8
        r, g, b = img[:, 0], img[:, 1], img[:, 2]

        # I corresponde ao maior valor RGB, como o V do HSV.
        value, max_idx = img.max(dim=1)
        img_min = img.min(dim=1).values
        delta = value - img_min

        # Calcula Hue por regioes, evitando divisao por zero.
        safe_delta = delta + eps
        hue_r = torch.remainder((g - b) / safe_delta, 6.0)
        hue_g = 2.0 + (b - r) / safe_delta
        hue_b = 4.0 + (r - g) / safe_delta

        hue = torch.where(max_idx == 0, hue_r, torch.zeros_like(value))
        hue = torch.where(max_idx == 1, hue_g, hue)
        hue = torch.where(max_idx == 2, hue_b, hue)
        hue = torch.where(delta <= eps, torch.zeros_like(hue), hue)
        hue = torch.remainder(hue / 6.0, 1.0)

        # Saturacao do HSV.
        saturation = delta / (value + eps)
        saturation = torch.where(value <= eps, torch.zeros_like(saturation), saturation)

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        intensity = value.unsqueeze(1)

        # Sensibilidade de cor dependente da intensidade.
        k = self.current_k().to(device=img.device, dtype=img.dtype)
        color_sensitive = (torch.sin(intensity * 0.5 * math.pi) + eps).pow(k)

        # Hue polarizado em dois eixos: Horizontal e Vertical.
        angle = 2.0 * math.pi * hue
        h_channel = color_sensitive * saturation * torch.cos(angle)
        v_channel = color_sensitive * saturation * torch.sin(angle)

        return torch.cat([h_channel, v_channel, intensity], dim=1)

    def hvi_to_rgb(self, hvi):
        eps = 1e-8
        h_channel = hvi[:, 0].clamp(-1.0, 1.0)
        v_channel = hvi[:, 1].clamp(-1.0, 1.0)
        intensity = hvi[:, 2].clamp(0.0, 1.0)

        # Remove a modulacao de intensidade usada na transformacao direta.
        k = self.current_k().to(device=hvi.device, dtype=hvi.dtype)
        color_sensitive = (torch.sin(intensity * 0.5 * math.pi) + eps).pow(k)
        h_norm = h_channel / (color_sensitive + eps)
        v_norm = v_channel / (color_sensitive + eps)
        h_norm = h_norm.clamp(-1.0, 1.0)
        v_norm = v_norm.clamp(-1.0, 1.0)

        # Recupera Hue e Saturation.
        hue = torch.remainder(torch.atan2(v_norm + eps, h_norm + eps) / (2.0 * math.pi), 1.0)
        saturation = torch.sqrt(h_norm.pow(2) + v_norm.pow(2) + eps).clamp(0.0, 1.0)
        value = intensity

        # Conversao HSV -> RGB vetorizada.
        h6 = hue * 6.0
        sector = torch.floor(h6).long() % 6
        fraction = h6 - torch.floor(h6)
        p = value * (1.0 - saturation)
        q = value * (1.0 - fraction * saturation)
        t = value * (1.0 - (1.0 - fraction) * saturation)

        r = torch.zeros_like(value)
        g = torch.zeros_like(value)
        b = torch.zeros_like(value)

        masks = [sector == i for i in range(6)]
        r = torch.where(masks[0], value, r)
        g = torch.where(masks[0], t, g)
        b = torch.where(masks[0], p, b)

        r = torch.where(masks[1], q, r)
        g = torch.where(masks[1], value, g)
        b = torch.where(masks[1], p, b)

        r = torch.where(masks[2], p, r)
        g = torch.where(masks[2], value, g)
        b = torch.where(masks[2], t, b)

        r = torch.where(masks[3], p, r)
        g = torch.where(masks[3], q, g)
        b = torch.where(masks[3], value, b)

        r = torch.where(masks[4], t, r)
        g = torch.where(masks[4], p, g)
        b = torch.where(masks[4], value, b)

        r = torch.where(masks[5], value, r)
        g = torch.where(masks[5], p, g)
        b = torch.where(masks[5], q, b)

        return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)



# LayerNorm 2D do Restormer: normaliza os canais em cada posicao espacial.

class LayerNorm2D(nn.Module):
    def __init__(self, dim, layernorm_type="WithBias"):
        super().__init__()
        if layernorm_type not in {"WithBias", "BiasFree"}:
            raise ValueError("layernorm_type deve ser 'WithBias' ou 'BiasFree'.")
        self.layernorm_type = layernorm_type
        self.weight = nn.Parameter(torch.ones(dim))
        if layernorm_type == "WithBias":
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.layernorm_type == "WithBias":
            mean = y.mean(dim=-1, keepdim=True)
            var = y.var(dim=-1, keepdim=True, unbiased=False)
            y = (y - mean) / torch.sqrt(var + 1e-5)
            y = y * self.weight + self.bias
        else:
            var = y.var(dim=-1, keepdim=True, unbiased=False)
            y = y / torch.sqrt(var + 1e-5)
            y = y * self.weight
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()



# MDTA do Restormer: atencao transposta entre canais, sem janelas espaciais.
# Isso evita a grade fixa de window_size usada pelo Swin.

class RestormerAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim precisa ser divisivel por num_heads.")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads

        # [B,C,H,W] -> [B,heads,C/head,HW]
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        # Normaliza Q e K antes da atencao, como no Restormer.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)



# GDFN do Restormer: feed-forward com convolucao depthwise e gating.

class RestormerFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)



# Bloco Restormer completo: LN -> MDTA -> residual -> LN -> GDFN -> residual.

class RestormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        expansion_factor=2.66,
        bias=False,
        layernorm_type="WithBias",
    ):
        super().__init__()
        self.norm1 = LayerNorm2D(dim, layernorm_type)
        self.attn = RestormerAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2D(dim, layernorm_type)
        self.ffn = RestormerFeedForward(dim, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



# Haar DWT somente para extracao de informacao de frequencia da intensidade.
# A DWT nao reconstrui diretamente a imagem final.

class HaarDWT(nn.Module):
    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh



# Ramo Wavelet de intensidade.
# As quatro bandas viram features auxiliares e depois voltam a HxW por
# interpolacao bilinear. Elas nao sao somadas diretamente ao RGB final.

class IntensityWaveletBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = HaarDWT()
        self.band_fusion = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, intensity):
        h0, w0 = intensity.shape[-2:]
        pad_h = h0 % 2
        pad_w = w0 % 2
        if pad_h or pad_w:
            mode = "reflect" if h0 > 1 and w0 > 1 else "replicate"
            intensity = F.pad(intensity, (0, pad_w, 0, pad_h), mode=mode)

        ll, lh, hl, hh = self.dwt(intensity)
        wave = self.band_fusion(torch.cat([ll, lh, hl, hh], dim=1))
        wave = F.interpolate(wave, size=intensity.shape[-2:], mode="bilinear", align_corners=False)
        wave = self.refine(wave)
        return wave[:, :, :h0, :w0]



# Fusao controlada entre os ramos de cor e intensidade.
# As convolucoes de troca comecam em zero, entao a arquitetura inicia com
# os dois ramos praticamente desacoplados e aprende a interacao gradualmente.

class CrossBranchFusion(nn.Module):
    def __init__(self, channels, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.intensity_to_color = nn.Conv2d(channels, channels, 1)
        self.color_to_intensity = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.intensity_to_color.weight)
        nn.init.zeros_(self.intensity_to_color.bias)
        nn.init.zeros_(self.color_to_intensity.weight)
        nn.init.zeros_(self.color_to_intensity.bias)

    def forward(self, color_feat, intensity_feat):
        color_update = self.intensity_to_color(intensity_feat)
        intensity_update = self.color_to_intensity(color_feat)
        color_feat = color_feat + self.scale * color_update
        intensity_feat = intensity_feat + self.scale * intensity_update
        return color_feat, intensity_feat



# Arquitetura principal.

class LCWHVINet(nn.Module):
    def __init__(
        self,
        channels=48,
        num_heads=4,
        depth=4,
        expansion_factor=2.66,
        wavelet_mode="on",
        color_mode="lock",
        color_scale=0.03,
        curve_steps=4,
        curve_scale=1.0,
        hvi_k=0.2,
        learnable_hvi_k=False,
        layernorm_type="WithBias",
        bias=False,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels precisa ser divisivel por num_heads.")
        if wavelet_mode not in {"on", "off"}:
            raise ValueError("wavelet_mode deve ser 'on' ou 'off'.")
        if color_mode not in {"lock", "bounded"}:
            raise ValueError("color_mode deve ser 'lock' ou 'bounded'.")
        if curve_steps <= 0:
            raise ValueError("curve_steps precisa ser maior que zero.")

        self.wavelet_mode = wavelet_mode
        self.use_wavelet = wavelet_mode == "on"
        self.color_mode = color_mode
        self.color_scale = float(color_scale)
        self.curve_steps = int(curve_steps)
        self.curve_scale = float(curve_scale)

        # Converte RGB <-> HVI.
        self.hvi = HVITransform(density_k=hvi_k, learnable_k=learnable_hvi_k)

        # Embedding da cor: recebe H e V.
        self.color_embed = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=bias),
            nn.GELU(),
        )

        # Embedding da intensidade: recebe somente I.
        self.intensity_embed = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=bias),
            nn.GELU(),
        )

        # Wavelet atua somente sobre a intensidade.
        self.wavelet = IntensityWaveletBranch(channels)
        self.wavelet_scale = 0.1

        # Dois backbones independentes de Restormer.
        self.color_body = nn.Sequential(*[
            RestormerBlock(
                channels,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                bias=bias,
                layernorm_type=layernorm_type,
            )
            for _ in range(depth)
        ])
        self.intensity_body = nn.Sequential(*[
            RestormerBlock(
                channels,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                bias=bias,
                layernorm_type=layernorm_type,
            )
            for _ in range(depth)
        ])

        # Troca controlada de informacao entre cor e intensidade.
        self.cross_fusion = CrossBranchFusion(channels, scale=0.1)

        # A cabeca de intensidade preve uma curva escalar por pixel.
        self.intensity_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1, bias=True),
        )

        # A cabeca de cor preve somente uma pequena correcao em H e V.
        self.color_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, 2, 3, padding=1, bias=True),
        )

        # Inicializacao em zero: a rede comeca como identidade.
        nn.init.zeros_(self.intensity_head[-1].weight)
        nn.init.zeros_(self.intensity_head[-1].bias)
        nn.init.zeros_(self.color_head[-1].weight)
        nn.init.zeros_(self.color_head[-1].bias)

    def current_hvi_k(self):
        return self.hvi.current_k()

    def apply_intensity_curve(self, intensity, raw_curve):
        # tanh limita a curva. Com raw_curve=0, a transformacao e identidade.
        curve = self.curve_scale * torch.tanh(raw_curve)
        enhanced = intensity
        for _ in range(self.curve_steps):
            enhanced = enhanced + curve * enhanced * (1.0 - enhanced)
            enhanced = enhanced.clamp(0.0, 1.0)
        return enhanced, curve

    def forward(self, img, return_aux=False):
        # A nova arquitetura trabalha exclusivamente em [0,1].
        img = img.clamp(0.0, 1.0)

        # 1. RGB -> HVI.
        hvi_in = self.hvi.rgb_to_hvi(img)
        hv_in = hvi_in[:, 0:2]
        intensity_in = hvi_in[:, 2:3]

        # 2. Extrai features independentes de cor e intensidade.
        color_feat = self.color_embed(hv_in)
        intensity_feat = self.intensity_embed(intensity_in)

        # 3. Wavelet adiciona frequencia somente ao ramo de intensidade.
        if self.use_wavelet:
            wave_feat = self.wavelet(intensity_in)
        elif self.training:
            # Mantem o ramo Wavelet no grafo quando DDP usa find_unused_parameters=False.
            wave_feat = self.wavelet(intensity_in) * 0.0
        else:
            wave_feat = torch.zeros_like(intensity_feat)
        intensity_feat = intensity_feat + self.wavelet_scale * wave_feat

        # 4. Processamento profundo com Restormer, sem janelas espaciais.
        color_feat = self.color_body(color_feat)
        intensity_feat = self.intensity_body(intensity_feat)

        # 5. Interacao controlada entre os dois ramos.
        color_feat, intensity_feat = self.cross_fusion(color_feat, intensity_feat)

        # 6. Corrige intensidade por uma curva limitada e iniciada em identidade.
        raw_curve = self.intensity_head(intensity_feat)
        intensity_out, curve = self.apply_intensity_curve(intensity_in, raw_curve)

        # 7. Cor: travada na fase inicial ou corrigida por delta pequeno e limitado.
        raw_delta_hv = self.color_head(color_feat)
        if self.color_mode == "lock":
            delta_hv = raw_delta_hv * 0.0
        else:
            delta_hv = self.color_scale * torch.tanh(raw_delta_hv)
        hv_out = (hv_in + delta_hv).clamp(-1.0, 1.0)

        # 8. Junta H, V e I corrigidos e volta para RGB.
        hvi_out = torch.cat([hv_out, intensity_out], dim=1)
        output = self.hvi.hvi_to_rgb(hvi_out).clamp(0.0, 1.0)

        # Valores internos para validacao e diagnostico.
        if not self.training:
            self.debug_hvi_in = hvi_in.detach()
            self.debug_hvi_out = hvi_out.detach()
            self.debug_hv_in = hv_in.detach()
            self.debug_hv_out = hv_out.detach()
            self.debug_intensity_in = intensity_in.detach()
            self.debug_intensity_out = intensity_out.detach()
            self.debug_curve = curve.detach()
            self.debug_delta_hv = delta_hv.detach()

        # Durante o treinamento, retorna variaveis auxiliares para losses HVI.
        if return_aux:
            aux = {
                "hvi_in": hvi_in,
                "hvi_out": hvi_out,
                "hv_in": hv_in,
                "hv_out": hv_out,
                "intensity_in": intensity_in,
                "intensity_out": intensity_out,
                "curve": curve,
                "delta_hv": delta_hv,
            }
            return output, aux

        return output


if __name__ == "__main__":
    # Teste simples: antes do treinamento a rede deve ficar proxima da identidade.
    model = LCWHVINet(
        channels=48,
        num_heads=4,
        depth=4,
        wavelet_mode="on",
        color_mode="lock",
    ).eval()

    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        y, aux = model(x, return_aux=True)

    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Output min/max:", y.min().item(), y.max().item())
    print("Erro medio inicial:", (y - x).abs().mean().item())
    print("Erro maximo inicial:", (y - x).abs().max().item())
    print("HVI k:", model.current_hvi_k().item())
    print("Curve abs mean:", aux["curve"].abs().mean().item())
    print("Delta HV abs mean:", aux["delta_hv"].abs().mean().item())
