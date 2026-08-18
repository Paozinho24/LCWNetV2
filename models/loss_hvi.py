import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.lcw_hvi_backbone import HVITransform
except ImportError:
    from User.LCWNetV2.models.lcw_hvi_backbone import HVITransform


# Perda L1 da imagem RGB final.
class RGBReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# Perda de intensidade no canal I do HVI.
class IntensityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, intensity_out, intensity_target):
        return self.loss(intensity_out, intensity_target)


# Perda direta dos canais cromaticos H e V.
class HVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, hv_out, hv_target):
        return self.loss(hv_out, hv_target)


# Controla a magnitude cromatica para evitar saturacao excessiva.
class ChromaMagnitudeLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, hv_out, hv_target):
        chroma_out = torch.sqrt(hv_out[:, 0:1].pow(2) + hv_out[:, 1:2].pow(2) + self.eps)
        chroma_target = torch.sqrt(hv_target[:, 0:1].pow(2) + hv_target[:, 1:2].pow(2) + self.eps)
        return F.l1_loss(chroma_out, chroma_target)


# Penaliza mudancas de direcao da cor no plano H-V.
class HueDirectionLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, hv_out, hv_target):
        out_norm = torch.sqrt(hv_out.pow(2).sum(dim=1, keepdim=True) + self.eps)
        target_norm = torch.sqrt(hv_target.pow(2).sum(dim=1, keepdim=True) + self.eps)

        hv_out_unit = hv_out / out_norm
        hv_target_unit = hv_target / target_norm

        cosine = (hv_out_unit * hv_target_unit).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

        # Pixels quase cinza possuem hue pouco confiavel e recebem peso menor.
        target_chroma = target_norm.clamp(0.0, 1.0)
        return (target_chroma * (1.0 - cosine)).mean()


# Preserva bordas da intensidade sem usar gradientes RGB independentes.
class IntensityGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, intensity_out, intensity_target):
        out_dx = intensity_out[:, :, :, 1:] - intensity_out[:, :, :, :-1]
        gt_dx = intensity_target[:, :, :, 1:] - intensity_target[:, :, :, :-1]
        out_dy = intensity_out[:, :, 1:, :] - intensity_out[:, :, :-1, :]
        gt_dy = intensity_target[:, :, 1:, :] - intensity_target[:, :, :-1, :]
        return F.l1_loss(out_dx, gt_dx) + F.l1_loss(out_dy, gt_dy)


# Suaviza o mapa de curva em regioes planas e preserva variacao perto de bordas reais.
class EdgeAwareCurveSmoothnessLoss(nn.Module):
    def __init__(self, edge_strength=10.0):
        super().__init__()
        self.edge_strength = float(edge_strength)

    def forward(self, curve, intensity_target):
        curve_dx = curve[:, :, :, 1:] - curve[:, :, :, :-1]
        curve_dy = curve[:, :, 1:, :] - curve[:, :, :-1, :]
        gt_dx = intensity_target[:, :, :, 1:] - intensity_target[:, :, :, :-1]
        gt_dy = intensity_target[:, :, 1:, :] - intensity_target[:, :, :-1, :]

        weight_x = torch.exp(-self.edge_strength * gt_dx.abs())
        weight_y = torch.exp(-self.edge_strength * gt_dy.abs())

        return (weight_x * curve_dx.abs()).mean() + (weight_y * curve_dy.abs()).mean()


# Mantem pequena a correcao cromatica aprendida no modo bounded.
class ColorDeltaRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_hv):
        return delta_hv.abs().mean()


# SSIM diferenciavel em RGB [0,1].
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    def _kernel(self, channels, device, dtype, window_size):
        coords = torch.arange(window_size, device=device, dtype=dtype)
        coords = coords - (window_size - 1) / 2.0
        gaussian = torch.exp(-(coords.pow(2)) / (2.0 * self.sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel_2d = gaussian[:, None] * gaussian[None, :]
        return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    def forward(self, pred, target):
        _, channels, height, width = pred.shape
        window_size = min(self.window_size, height, width)
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(1, window_size)

        kernel = self._kernel(channels, pred.device, pred.dtype, window_size)
        padding = window_size // 2

        mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
        mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_cross

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        ssim_map = numerator / denominator.clamp_min(1e-12)
        return 1.0 - ssim_map.mean()


# Loss total da LCW-HVI-Restormer.
class LCWHVITotalLoss(nn.Module):
    def __init__(
        self,
        rgb_weight=1.0,
        intensity_weight=0.5,
        hv_weight=0.5,
        chroma_weight=0.2,
        hue_weight=0.1,
        grad_weight=0.05,
        curve_smooth_weight=0.02,
        color_delta_weight=0.02,
        ssim_weight=0.1,
        color_mode="lock",
        hvi_k=0.2,
        edge_strength=10.0,
    ):
        super().__init__()

        if color_mode not in {"lock", "bounded"}:
            raise ValueError("color_mode deve ser 'lock' ou 'bounded'.")

        self.rgb_weight = float(rgb_weight)
        self.intensity_weight = float(intensity_weight)
        self.hv_weight = float(hv_weight)
        self.chroma_weight = float(chroma_weight)
        self.hue_weight = float(hue_weight)
        self.grad_weight = float(grad_weight)
        self.curve_smooth_weight = float(curve_smooth_weight)
        self.color_delta_weight = float(color_delta_weight)
        self.ssim_weight = float(ssim_weight)
        self.color_mode = color_mode

        # Usa a mesma transformacao HVI fixa do backbone para criar o GT em HVI.
        self.hvi = HVITransform(density_k=hvi_k, learnable_k=False)

        self.rgb_loss = RGBReconstructionLoss()
        self.intensity_loss = IntensityLoss()
        self.hv_loss = HVLoss()
        self.chroma_loss = ChromaMagnitudeLoss()
        self.hue_loss = HueDirectionLoss()
        self.grad_loss = IntensityGradientLoss()
        self.curve_smooth_loss = EdgeAwareCurveSmoothnessLoss(edge_strength=edge_strength)
        self.color_delta_loss = ColorDeltaRegularization()
        self.ssim_loss = SSIMLoss()

    def set_color_mode(self, color_mode):
        if color_mode not in {"lock", "bounded"}:
            raise ValueError("color_mode deve ser 'lock' ou 'bounded'.")
        self.color_mode = color_mode

    def forward(self, output, target, aux):
        if aux is None:
            raise ValueError("aux e obrigatorio. Use model(input_img, return_aux=True).")

        required_keys = {
            "hv_out",
            "intensity_out",
            "curve",
            "delta_hv",
        }
        missing = required_keys.difference(aux.keys())
        if missing:
            raise KeyError(f"Chaves ausentes em aux: {sorted(missing)}")

        # Toda a nova arquitetura trabalha em RGB [0,1].
        output = output.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        # Converte o GT para HVI para supervisionar intensidade e cor separadamente.
        target_hvi = self.hvi.rgb_to_hvi(target)
        hv_target = target_hvi[:, 0:2]
        intensity_target = target_hvi[:, 2:3]

        # Reconstrucao RGB final.
        loss_rgb = self.rgb_loss(output, target)

        # Supervisiona somente a intensidade no ramo de iluminacao.
        loss_intensity = self.intensity_loss(aux["intensity_out"], intensity_target)
        loss_grad = self.grad_loss(aux["intensity_out"], intensity_target)
        loss_curve_smooth = self.curve_smooth_loss(aux["curve"], intensity_target)

        # SSIM atua na imagem RGB final para preservar estrutura global/local.
        if self.ssim_weight > 0.0:
            loss_ssim = self.ssim_loss(output, target)
        else:
            loss_ssim = output.new_tensor(0.0)

        # No modo lock a cor fica congelada. As losses cromaticas sao apenas medidas.
        loss_hv = self.hv_loss(aux["hv_out"], hv_target)
        loss_chroma = self.chroma_loss(aux["hv_out"], hv_target)
        loss_hue = self.hue_loss(aux["hv_out"], hv_target)
        loss_color_delta = self.color_delta_loss(aux["delta_hv"])

        color_trainable = self.color_mode == "bounded"
        color_factor = 1.0 if color_trainable else 0.0

        total = (
            self.rgb_weight * loss_rgb
            + self.intensity_weight * loss_intensity
            + self.grad_weight * loss_grad
            + self.curve_smooth_weight * loss_curve_smooth
            + self.ssim_weight * loss_ssim
            + color_factor * self.hv_weight * loss_hv
            + color_factor * self.chroma_weight * loss_chroma
            + color_factor * self.hue_weight * loss_hue
            + color_factor * self.color_delta_weight * loss_color_delta
        )

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_rgb": float(loss_rgb.detach().cpu()),
            "loss_intensity": float(loss_intensity.detach().cpu()),
            "loss_hv": float(loss_hv.detach().cpu()),
            "loss_chroma": float(loss_chroma.detach().cpu()),
            "loss_hue": float(loss_hue.detach().cpu()),
            "loss_grad": float(loss_grad.detach().cpu()),
            "loss_curve_smooth": float(loss_curve_smooth.detach().cpu()),
            "loss_color_delta": float(loss_color_delta.detach().cpu()),
            "loss_ssim": float(loss_ssim.detach().cpu()),
            "color_trainable": int(color_trainable),
        }

        return total, logs


if __name__ == "__main__":
    from User.LCWNetV2.models.lcw_hvi_backbone import LCWHVINet

    model = LCWHVINet(
        channels=24,
        num_heads=4,
        depth=2,
        wavelet_mode="on",
        color_mode="lock",
    )

    criterion = LCWHVITotalLoss(
        color_mode="lock",
        hvi_k=0.2,
    )

    low = torch.rand(2, 3, 64, 64)
    gt = torch.rand(2, 3, 64, 64)

    output, aux = model(low, return_aux=True)
    loss, logs = criterion(output, gt, aux)
    loss.backward()

    print("Loss:", loss.item())
    for key, value in logs.items():
        print(f"{key}: {value}")
