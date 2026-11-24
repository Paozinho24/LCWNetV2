
from __future__ import annotations
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.lcw_hvi_backbone import (
    HVITransform,
    IntensityWaveletBranch,
    DeformableResidualBlock,
    RestormerBlock,
    CrossBranchFusion,
)

class Downsample(nn.Module):
    """Convolucao de stride=2, preservando suporte a tamanhos impares."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.PReLU(num_parameters=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleWithSkip(nn.Module):
    """Upsampling + concatenacao + projecao para a largura do encoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class LCWHVIUNet(nn.Module):
    """UNet de dois ramos, com 4 escalas e 6 estagios de fusao bidirecional.

    Escalas: H, H/2, H/4, H/8. Restormer em ambos os ramos.
    Heads: curvas iterativas e residual de I; delta cromatico normalizado.
    Assinatura e dicionario aux compativeis com a loss da branch E1.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 64, 128, 192),
        num_heads: Sequence[int] = (4, 4, 4, 4),
        encoder_depths: Sequence[int] = (1, 1, 1, 2),
        decoder_depth: int = 1,
        expansion_factor: float = 2.66,
        wavelet_mode: str = "on",
        color_mode: str = "lock",
        color_scale: float = 0.05,
        curve_steps: int = 6,
        curve_scale: float = 1.0,
        hvi_k: float = 0.2,
        learnable_hvi_k: bool = False,
        layernorm_type: str = "WithBias",
        bias: bool = False,
        fusion_scale: float = 0.2,
        wavelet_scale: float = 0.2,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if len(channels) != 4 or len(num_heads) != 4 or len(encoder_depths) != 4:
            raise ValueError("channels, num_heads e encoder_depths devem ter 4 elementos")
        if any(c <= 0 or h <= 0 or c % h for c, h in zip(channels, num_heads)):
            raise ValueError("Cada largura de canais deve ser positiva e divisivel por num_heads")
        if min(encoder_depths) <= 0 or decoder_depth <= 0 or curve_steps <= 0:
            raise ValueError("Profundidades e curve_steps precisam ser positivos")
        if wavelet_mode not in {"on", "off"}:
            raise ValueError("wavelet_mode deve ser 'on' ou 'off'")
        if color_mode not in {"lock", "bounded"}:
            raise ValueError("color_mode deve ser 'lock' ou 'bounded'")
        if color_scale < 0 or curve_scale <= 0 or residual_scale < 0:
            raise ValueError("Escalas fora dos limites validos")

        c0, c1, c2, c3 = map(int, channels)
        self.channels = tuple(channels)
        self.curve_steps = int(curve_steps)
        self.curve_scale = float(curve_scale)
        self.color_mode = color_mode
        self.color_scale = float(color_scale)
        self.residual_scale = float(residual_scale)
        self.wavelet_mode = wavelet_mode
        self.use_wavelet = wavelet_mode == "on"
        self.wavelet_scale = float(wavelet_scale)
        self.hvi = HVITransform(density_k=hvi_k, learnable_k=learnable_hvi_k)

        self.color_embed = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1, bias=bias), nn.PReLU(num_parameters=c0)
        )
        self.intensity_embed = nn.Sequential(
            nn.Conv2d(1, c0, 3, padding=1, bias=bias), nn.PReLU(num_parameters=c0)
        )
        self.wavelet = IntensityWaveletBranch(c0)
        self.intensity_deform = DeformableResidualBlock(c0)

        def stage(width: int, heads: int, depth: int) -> nn.Sequential:
            return nn.Sequential(*[
                RestormerBlock(
                    width,
                    num_heads=heads,
                    expansion_factor=expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                )
                for _ in range(depth)
            ])

        # Dois encoders independentes com as mesmas resolucoes.
        self.color_enc = nn.ModuleList([
            stage(c, h, d) for c, h, d in zip(channels, num_heads, encoder_depths)
        ])
        self.intensity_enc = nn.ModuleList([
            stage(c, h, d) for c, h, d in zip(channels, num_heads, encoder_depths)
        ])
        self.color_down = nn.ModuleList([
            Downsample(a, b) for a, b in zip(channels[:-1], channels[1:])
        ])
        self.intensity_down = nn.ModuleList([
            Downsample(a, b) for a, b in zip(channels[:-1], channels[1:])
        ])

        # Contexto global apenas no bottleneck: inicializacao neutra.
        self.intensity_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c3, c3, 1),
            nn.PReLU(num_parameters=c3),
            nn.Conv2d(c3, c3, 1),
        )
        nn.init.zeros_(self.intensity_global[-1].weight)
        nn.init.zeros_(self.intensity_global[-1].bias)

        # Duas fusoes no bottleneck; 2 no encoder e 2 no decoder = seis.
        # Mantem a SEMANTICA da CrossBranchFusion E1 (nao implementa LCA).
        self.cross_fusions = nn.ModuleList([
            CrossBranchFusion(c1, scale=fusion_scale),  # encoder H/2
            CrossBranchFusion(c2, scale=fusion_scale),  # encoder H/4
            CrossBranchFusion(c3, scale=fusion_scale),  # bottleneck 1
            CrossBranchFusion(c3, scale=fusion_scale),  # bottleneck 2
            CrossBranchFusion(c2, scale=fusion_scale),  # decoder H/4
            CrossBranchFusion(c1, scale=fusion_scale),  # decoder H/2
        ])

        # Decoders: indice 0 = H/4, 1 = H/2, 2 = H.
        self.color_up = nn.ModuleList([
            UpsampleWithSkip(c3, c2, c2),
            UpsampleWithSkip(c2, c1, c1),
            UpsampleWithSkip(c1, c0, c0),
        ])
        self.intensity_up = nn.ModuleList([
            UpsampleWithSkip(c3, c2, c2),
            UpsampleWithSkip(c2, c1, c1),
            UpsampleWithSkip(c1, c0, c0),
        ])
        self.color_dec = nn.ModuleList([
            stage(c2, num_heads[2], decoder_depth),
            stage(c1, num_heads[1], decoder_depth),
            stage(c0, num_heads[0], decoder_depth),
        ])
        self.intensity_dec = nn.ModuleList([
            stage(c2, num_heads[2], decoder_depth),
            stage(c1, num_heads[1], decoder_depth),
            stage(c0, num_heads[0], decoder_depth),
        ])

        # Mesma familia de cabecas da LCWHVINet E1.
        self.intensity_head = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=bias),
            nn.PReLU(num_parameters=c0),
            nn.Conv2d(c0, self.curve_steps, 3, padding=1, bias=True),
        )
        self.intensity_residual_head = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=bias),
            nn.PReLU(num_parameters=c0),
            nn.Conv2d(c0, 1, 3, padding=1, bias=True),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=bias),
            nn.PReLU(num_parameters=c0),
            nn.Conv2d(c0, 2, 3, padding=1, bias=True),
        )
        for head in (self.intensity_head, self.intensity_residual_head, self.color_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def current_hvi_k(self) -> torch.Tensor:
        return self.hvi.current_k()

    def apply_intensity_curve(
        self, intensity: torch.Tensor, raw_curve: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        curves = self.curve_scale * torch.tanh(raw_curve)
        enhanced = intensity
        for j in range(self.curve_steps):
            a = curves[:, j:j+1, :, :]
            enhanced = enhanced + a * enhanced * (1.0 - enhanced)
        return enhanced, curves

    def forward(self, img: torch.Tensor, return_aux: bool = False):
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError("Entrada esperada: tensor RGB [B,3,H,W]")
        if min(img.shape[-2:]) < 8:
            raise ValueError("Altura e largura devem ser >= 8 para tres downsamplings")
        img = img.clamp(0.0, 1.0)
        hvi_in = self.hvi.rgb_to_hvi(img)
        hv_in, intensity_in = hvi_in[:, :2], hvi_in[:, 2:3]

        color_feat = self.color_embed(torch.cat([hv_in, intensity_in], dim=1))
        intensity_feat = self.intensity_embed(intensity_in)

        if self.use_wavelet:
            wave_feat = self.wavelet(intensity_in)
        elif self.training:
            # Conecta os parametros wavelet ao grafo, mesmo quando desligado.
            wave_feat = 0.0 * self.wavelet(intensity_in)
        else:
            wave_feat = torch.zeros_like(intensity_feat)
        intensity_feat = intensity_feat + self.wavelet_scale * wave_feat
        intensity_feat = self.intensity_deform(intensity_feat)

        color_skips, intensity_skips = [], []
        for level in range(4):
            color_feat = self.color_enc[level](color_feat)
            intensity_feat = self.intensity_enc[level](intensity_feat)
            if level == 3:
                intensity_feat = intensity_feat + self.intensity_global(intensity_feat)
            if level > 0:
                color_feat, intensity_feat = self.cross_fusions[level - 1](
                    color_feat, intensity_feat
                )
            if level < 3:
                color_skips.append(color_feat)
                intensity_skips.append(intensity_feat)
                color_feat = self.color_down[level](color_feat)
                intensity_feat = self.intensity_down[level](intensity_feat)

        # Segundo estagio de comunicacao no bottleneck.
        color_feat, intensity_feat = self.cross_fusions[3](color_feat, intensity_feat)

        for j, level in enumerate((2, 1, 0)):
            color_feat = self.color_up[j](color_feat, color_skips[level])
            intensity_feat = self.intensity_up[j](intensity_feat, intensity_skips[level])
            color_feat = self.color_dec[j](color_feat)
            intensity_feat = self.intensity_dec[j](intensity_feat)
            if level > 0:
                # j=0 -> fusao 4 (H/4), j=1 -> fusao 5 (H/2).
                color_feat, intensity_feat = self.cross_fusions[4+j](
                    color_feat, intensity_feat
                )

        raw_curve = self.intensity_head(intensity_feat)
        intensity_curve, curve = self.apply_intensity_curve(intensity_in, raw_curve)
        residual_i = self.residual_scale * torch.tanh(self.intensity_residual_head(intensity_feat))
        intensity_out = (intensity_curve + residual_i).clamp(0.0, 1.0)

        # Preserva a reconstrucao cromatica normalizada da branch E1.
        color_sensitivity_in = self.hvi.color_sensitive_from_intensity(intensity_in)
        hv_normalized_in = self.hvi.limit_chroma_vector(
            hv_in / (color_sensitivity_in + 1e-8)
        )
        raw_delta_hv_normalized = self.color_head(color_feat)
        if self.color_mode == "lock":
            delta_hv_normalized = raw_delta_hv_normalized * 0.0
        else:
            delta_hv_normalized = self.color_scale * torch.tanh(raw_delta_hv_normalized)
        hv_normalized_out = self.hvi.limit_chroma_vector(
            hv_normalized_in + delta_hv_normalized
        )
        color_sensitivity_out = self.hvi.color_sensitive_from_intensity(intensity_out)
        hv_out = (hv_normalized_out * color_sensitivity_out).clamp(-1.0, 1.0)
        delta_hv = hv_out - hv_in

        hvi_out = torch.cat([hv_out, intensity_out], dim=1)
        output = self.hvi.hvi_to_rgb(hvi_out).clamp(0.0, 1.0)
        if not self.training:
            self.debug_hvi_in = hvi_in.detach()
            self.debug_hvi_out = hvi_out.detach()
            self.debug_hv_in = hv_in.detach()
            self.debug_hv_out = hv_out.detach()
            self.debug_intensity_in = intensity_in.detach()
            self.debug_intensity_out = intensity_out.detach()
            self.debug_curve = curve.detach()
            self.debug_delta_hv = delta_hv.detach()

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
                "delta_hv_normalized": delta_hv_normalized,
                "hv_normalized_in": hv_normalized_in,
                "hv_normalized_out": hv_normalized_out,
            }
            return output, aux
        return output


if __name__ == "__main__":
    model = LCWHVIUNet().eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 128, 128)
        y, aux = model(x, return_aux=True)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Inicializacao |pred-input|:", (y-x).abs().mean().item())
    print("Curvas:", tuple(aux["curve"].shape))
