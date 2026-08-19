import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from models.lcw_hvi_backbone import LCWHVINet
from dataload.llie_dataset import DATASET_CHOICES, IMAGE_EXTENSIONS, resolve_dataset_paths

ARCHITECTURE_VERSION = "LCWHVINet_HVI_Restormer_v1"


# Argumentos da inferência.
def get_args():
    parser = argparse.ArgumentParser(
        description="Inferência unificada da LCWHVINet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint e dataset.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/unicornio/User/DataSetsLLIE",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lsd",
        choices=DATASET_CHOICES,
    )
    parser.add_argument("--val_subset", type=str, default="DEI")

    # Se input_path estiver vazio, usa automaticamente a validação do dataset.
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/unicornio/User/LCWNet/results/LCWHVINet",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--suffix", type=str, default="_LCWHVI")

    # Inferência por tiles.
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="zero processa a imagem inteira",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--tile_log_interval",
        type=int,
        default=25,
    )

    # Execução.
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="ativa autocast FP16; deixe desligado nos primeiros testes de cor",
    )

    # Modo cromático salvo no checkpoint é usado por padrão.
    parser.add_argument(
        "--color_mode",
        type=str,
        default="auto",
        choices=["auto", "lock", "bounded"],
    )

    # Métricas.
    parser.add_argument(
        "--disable_lpips",
        action="store_true",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
    )
    parser.add_argument(
        "--lpips_max_size",
        type=int,
        default=256,
    )

    return parser.parse_args()


def validate_args(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nenhuma GPU CUDA está disponível.")

    if args.tile_size < 0:
        raise ValueError("tile_size não pode ser negativo.")

    if args.tile_overlap < 0:
        raise ValueError("tile_overlap não pode ser negativo.")

    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        raise ValueError("tile_overlap precisa ser menor que tile_size.")

    if args.lpips_max_size < 0:
        raise ValueError("lpips_max_size não pode ser negativo.")


# Carrega checkpoint completo.
def load_checkpoint_file(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict(checkpoint):
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint não contém model_state_dict.")

    state_dict = {}

    for name, tensor in checkpoint["model_state_dict"].items():
        if name.startswith("module."):
            name = name[len("module."):]
        state_dict[name] = tensor

    return state_dict


# Reconstrói a arquitetura usando a configuração salva no treinamento.
def load_model(args, device):
    checkpoint = load_checkpoint_file(args.checkpoint)

    version = checkpoint.get("architecture_version", "")

    if version != ARCHITECTURE_VERSION:
        raise RuntimeError(
            "Checkpoint incompatível com a LCWHVINet atual. "
            f"Esperado={ARCHITECTURE_VERSION!r}; "
            f"encontrado={version or 'não registrado'!r}."
        )

    config = checkpoint.get("config", {})

    if isinstance(config, argparse.Namespace):
        config = vars(config)

    if not isinstance(config, dict):
        config = {}

    saved_color_mode = str(
        checkpoint.get(
            "active_color_mode",
            config.get("color_mode", "lock"),
        )
    ).lower()

    if saved_color_mode not in {"lock", "bounded"}:
        raise ValueError(
            f"color_mode inválido no checkpoint: {saved_color_mode!r}"
        )

    color_mode = (
        saved_color_mode
        if args.color_mode == "auto"
        else args.color_mode
    )

    model_config = {
        "channels": int(config.get("channels", 48)),
        "num_heads": int(config.get("num_heads", 4)),
        "depth": int(config.get("depth", 4)),
        "expansion_factor": float(config.get("expansion_factor", 2.66)),
        "wavelet_mode": str(config.get("wavelet_mode", "on")),
        "color_mode": color_mode,
        "color_scale": float(config.get("color_scale", 0.03)),
        "curve_steps": int(config.get("curve_steps", 4)),
        "curve_scale": float(config.get("curve_scale", 1.0)),
        "hvi_k": float(config.get("hvi_k", 0.2)),
        "learnable_hvi_k": False,
        "layernorm_type": str(config.get("layernorm_type", "WithBias")),
        "bias": bool(config.get("bias", False)),
    }

    model = LCWHVINet(**model_config)
    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()

    info = {
        **model_config,
        "saved_color_mode": saved_color_mode,
        "architecture_version": version,
        "epoch": int(checkpoint.get("epoch", -1)) + 1,
        "metrics": checkpoint.get("metrics", {}),
    }

    return model, info


# Localiza entrada e GT automaticamente ou usa caminhos informados.
def resolve_inference_paths(args):
    if args.input_path:
        input_path = Path(args.input_path)
        gt_path = Path(args.gt_path) if args.gt_path else None
        return input_path, gt_path

    paths = resolve_dataset_paths(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        val_subset=args.val_subset,
    )

    input_path = paths["val_low"]

    if args.gt_path:
        gt_path = Path(args.gt_path)
    else:
        gt_path = paths["val_gt"]

    return input_path, gt_path


# Lista imagens de arquivo ou diretório.
def list_images(path, recursive=False):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Caminho de entrada não encontrado: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Extensão não suportada: {path.suffix}")
        return [path]

    iterator = path.rglob("*") if recursive else path.glob("*")

    images = sorted(
        [
            item
            for item in iterator
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda item: str(item).lower(),
    )

    if not images:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {path}")

    return images


# Cria mapa de GT por nome-base.
def build_gt_map(gt_path, recursive=False):
    if gt_path is None:
        return {}

    gt_path = Path(gt_path)

    if gt_path.is_file():
        return {"__single_file__": gt_path}

    images = list_images(gt_path, recursive=recursive)
    gt_map = {}

    for path in images:
        key = path.stem.lower()

        if key in gt_map:
            raise RuntimeError(
                f"Nome-base GT duplicado: {key}. "
                "Evite nomes repetidos ou desative --recursive."
            )

        gt_map[key] = path

    return gt_map


# Gera nomes candidatos para pares LSD e LOL.
def candidate_gt_keys(image_path):
    stem = Path(image_path).stem.lower()
    candidates = [stem]

    replacements = (
        ("_low", "_gt"),
        ("-low", "-gt"),
        (" low", " gt"),
        ("_input", "_gt"),
        ("-input", "-gt"),
        ("_low", "_high"),
        ("-low", "-high"),
        ("_input", "_high"),
        ("-input", "-high"),
    )

    for source, target in replacements:
        if stem.endswith(source):
            candidates.append(stem[:-len(source)] + target)
            candidates.append(stem[:-len(source)])

    unique = []

    for item in candidates:
        if item not in unique:
            unique.append(item)

    return unique


def find_gt(image_path, gt_map):
    if not gt_map:
        return None

    if "__single_file__" in gt_map:
        return gt_map["__single_file__"]

    for key in candidate_gt_keys(image_path):
        if key in gt_map:
            return gt_map[key]

    return None


# Carrega RGB em float32 [0,1].
def load_rgb(path):
    with Image.open(path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        tensor = pil_to_tensor(image).float() / 255.0

    return tensor.unsqueeze(0).clamp(0.0, 1.0)


# Salva saída RGB em PNG.
def save_rgb(tensor, path):
    tensor = tensor.detach().float().clamp(0.0, 1.0)
    image = to_pil_image(tensor.squeeze(0).cpu())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def output_path_for(image_path, input_root, output_dir, suffix):
    image_path = Path(image_path)
    input_root = Path(input_root)
    output_dir = Path(output_dir)

    if input_root.is_dir():
        try:
            parent = image_path.relative_to(input_root).parent
        except ValueError:
            parent = Path()
    else:
        parent = Path()

    return output_dir / parent / f"{image_path.stem}{suffix}.png"


# Executa o modelo uma vez.
def forward_model(model, tensor):
    output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("A saída do modelo não é tensor.")

    return output.clamp(0.0, 1.0)


def tile_positions(size, tile_size, step):
    if size <= tile_size:
        return [0]

    positions = list(range(0, size - tile_size + 1, step))
    last = size - tile_size

    if positions[-1] != last:
        positions.append(last)

    return positions


# Janela suave reduz emendas entre tiles.
def blend_window(height, width, device):
    if height == 1:
        weight_h = torch.ones(1, device=device)
    else:
        weight_h = torch.hann_window(
            height, periodic=False, device=device
        )

    if width == 1:
        weight_w = torch.ones(1, device=device)
    else:
        weight_w = torch.hann_window(
            width, periodic=False, device=device
        )

    weight = weight_h[:, None] * weight_w[None, :]
    return weight.clamp_min(1e-3).unsqueeze(0).unsqueeze(0)


# Processa imagem inteira ou por tiles sobrepostos.
def infer_image(
    model,
    tensor,
    tile_size,
    tile_overlap,
    tile_log_interval,
):
    _, _, height, width = tensor.shape

    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return forward_model(model, tensor)

    step = tile_size - tile_overlap
    y_positions = tile_positions(height, tile_size, step)
    x_positions = tile_positions(width, tile_size, step)
    total_tiles = len(y_positions) * len(x_positions)

    output_sum = torch.zeros(
        (1, 3, height, width),
        device=tensor.device,
        dtype=torch.float32,
    )

    weight_sum = torch.zeros(
        (1, 1, height, width),
        device=tensor.device,
        dtype=torch.float32,
    )

    processed = 0

    print(
        f"  Tiles: {len(y_positions)} x {len(x_positions)} = {total_tiles}",
        flush=True,
    )

    for top in y_positions:
        bottom = min(top + tile_size, height)

        for left in x_positions:
            right = min(left + tile_size, width)

            tile = tensor[:, :, top:bottom, left:right]
            output_tile = forward_model(model, tile).float()

            tile_h = bottom - top
            tile_w = right - left
            weight = blend_window(tile_h, tile_w, tensor.device)

            output_sum[:, :, top:bottom, left:right] += output_tile * weight
            weight_sum[:, :, top:bottom, left:right] += weight

            processed += 1

            if tile_log_interval > 0 and (
                processed % tile_log_interval == 0
                or processed == total_tiles
            ):
                print(
                    f"  Tiles processados: {processed}/{total_tiles} "
                    f"({100.0 * processed / total_tiles:.1f}%)",
                    flush=True,
                )

    return (output_sum / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)


# PSNR em RGB [0,1].
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()

    if mse <= 1e-12:
        return 100.0

    return 10.0 * math.log10(1.0 / mse)


def gaussian_kernel(window_size, channels, device, dtype):
    sigma = 1.5
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


# SSIM em RGB [0,1].
def calculate_ssim(pred, target):
    _, channels, height, width = pred.shape
    size = min(11, height, width)

    if size % 2 == 0:
        size -= 1

    size = max(1, size)
    kernel = gaussian_kernel(size, channels, pred.device, pred.dtype)
    padding = size // 2

    mu1 = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred * pred, kernel, padding=padding, groups=channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * target, kernel, padding=padding, groups=channels)
        - mu12
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (
        sigma1_sq + sigma2_sq + c2
    )

    return float(
        (numerator / denominator.clamp_min(1e-12)).mean().item()
    )


# Métricas específicas para o problema de intensidade/cor.
def calculate_hvi_metrics(model, pred, target):
    hvi_pred = model.hvi.rgb_to_hvi(pred)
    hvi_target = model.hvi.rgb_to_hvi(target)

    hv_pred = hvi_pred[:, 0:2]
    hv_target = hvi_target[:, 0:2]

    chroma_pred = torch.sqrt(
        hv_pred[:, 0:1].pow(2)
        + hv_pred[:, 1:2].pow(2)
        + 1e-8
    )
    chroma_target = torch.sqrt(
        hv_target[:, 0:1].pow(2)
        + hv_target[:, 1:2].pow(2)
        + 1e-8
    )

    intensity_mae = F.l1_loss(
        hvi_pred[:, 2:3],
        hvi_target[:, 2:3],
    ).item()

    chroma_mae = F.l1_loss(
        chroma_pred,
        chroma_target,
    ).item()

    high_clip_fraction = (
        pred >= 0.999
    ).float().mean().item()

    return intensity_mae, chroma_mae, high_clip_fraction


def load_lpips(device, network):
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "LPIPS não está instalado. Instale com 'pip install lpips' "
            "ou execute com --disable_lpips."
        ) from error

    metric = lpips.LPIPS(net=network, version="0.1").to(device).eval()

    for parameter in metric.parameters():
        parameter.requires_grad_(False)

    return metric


def resize_for_lpips(tensor, max_size):
    if max_size <= 0:
        return tensor

    height, width = tensor.shape[-2:]
    largest = max(height, width)

    if largest <= max_size:
        return tensor

    scale = max_size / float(largest)
    new_h = max(32, int(round(height * scale)))
    new_w = max(32, int(round(width * scale)))

    return F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )


def calculate_lpips(pred, target, metric, max_size):
    pred = resize_for_lpips(pred, max_size)
    target = resize_for_lpips(target, max_size)

    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0

    value = metric(pred, target)
    return float(value.mean().detach().cpu())


def write_csv_header(path):
    fields = [
        "image",
        "gt",
        "width",
        "height",
        "seconds",
        "psnr",
        "ssim",
        "lpips",
        "intensity_mae",
        "chroma_mae",
        "high_clip_fraction",
        "output_mean",
        "output_path",
        "status",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fields)


def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def write_summary(path, summary):
    lines = [
        "RESUMO DA INFERÊNCIA LCWHVINet",
        "",
        f"Imagens processadas: {summary['images']}",
        f"Pares avaliados: {summary['pairs']}",
        f"Tempo total: {summary['total_time']:.6f} s",
        f"Tempo médio: {summary['mean_time']:.6f} s/imagem",
    ]

    if summary["pairs"] > 0:
        lines += [
            f"PSNR médio: {summary['psnr']:.6f} dB",
            f"SSIM médio: {summary['ssim']:.6f}",
            f"LPIPS médio: {summary['lpips']:.6f}"
            if summary["lpips"] is not None
            else "LPIPS médio: não calculado",
            f"Intensity MAE médio: {summary['intensity_mae']:.6f}",
            f"Chroma MAE médio: {summary['chroma_mae']:.6f}",
            f"Fração média de pixels >= 0.999: "
            f"{summary['high_clip_fraction']:.8f}",
        ]

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = get_args()
    validate_args(args)

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Carrega modelo e configuração do checkpoint.
    model, model_info = load_model(args, device)

    # 2. Resolve o conjunto de imagens.
    input_path, gt_path = resolve_inference_paths(args)
    image_paths = list_images(input_path, recursive=args.recursive)
    gt_map = build_gt_map(gt_path, recursive=args.recursive)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "metrics_summary.txt"
    write_csv_header(csv_path)

    lpips_metric = None

    if gt_map and not args.disable_lpips:
        lpips_metric = load_lpips(device, args.lpips_net)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("=" * 84)
    print("INFERÊNCIA LCWHVINet")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Época:               {model_info['epoch']}")
    print(f"Versão:              {model_info['architecture_version']}")
    print(f"Dataset:             {args.dataset_name}")
    print(f"Entrada:             {input_path}")
    print(f"GT:                  {gt_path or 'desativado'}")
    print(f"Saída:               {output_dir}")
    print(f"Imagens:             {len(image_paths)}")
    print(f"Device:              {device}")
    print(f"AMP:                 {args.amp and device.type == 'cuda'}")
    print(f"Tile/overlap:        {args.tile_size}/{args.tile_overlap}")
    print(f"Wavelet:             {model_info['wavelet_mode']}")
    print(f"Color mode salvo:    {model_info['saved_color_mode']}")
    print(f"Color mode usado:    {model_info['color_mode']}")
    print(f"Color scale:         {model_info['color_scale']}")
    print(f"Curve steps:         {model_info['curve_steps']}")
    print(f"Curve scale:         {model_info['curve_scale']}")
    print(f"HVI k:               {model_info['hvi_k']}")
    print(f"Channels:            {model_info['channels']}")
    print(f"Depth:               {model_info['depth']}")
    print(f"Heads:               {model_info['num_heads']}")
    print(f"Parâmetros:          {total_params / 1e6:.3f} M")
    print(f"Treináveis:          {trainable_params / 1e6:.3f} M")
    print(f"LPIPS:               {'desativado' if lpips_metric is None else args.lpips_net}")
    print("=" * 84)

    total_time = 0.0
    pair_count = 0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0
    lpips_count = 0
    sum_intensity = 0.0
    sum_chroma = 0.0
    sum_high_clip = 0.0

    autocast_enabled = args.amp and device.type == "cuda"

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            print(
                f"\n[{index}/{len(image_paths)}] {image_path.name}",
                flush=True,
            )

            input_tensor = load_rgb(image_path).to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start = time.time()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                output = infer_image(
                    model=model,
                    tensor=input_tensor,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    tile_log_interval=args.tile_log_interval,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.time() - start
            total_time += elapsed

            output_path = output_path_for(
                image_path,
                input_path,
                output_dir,
                args.suffix,
            )
            save_rgb(output, output_path)

            height, width = input_tensor.shape[-2:]
            gt_file = find_gt(image_path, gt_map)

            psnr = None
            ssim = None
            lpips_value = None
            intensity_mae = None
            chroma_mae = None
            high_clip = float(
                (output >= 0.999).float().mean().cpu()
            )
            status = "ok"

            if gt_map and gt_file is None:
                status = "gt_nao_encontrado"

            if gt_file is not None:
                target = load_rgb(gt_file).to(
                    device,
                    non_blocking=True,
                )

                if target.shape != output.shape:
                    status = (
                        f"dimensoes_diferentes:"
                        f"pred={tuple(output.shape)};"
                        f"gt={tuple(target.shape)}"
                    )
                else:
                    psnr = calculate_psnr(output.float(), target.float())
                    ssim = calculate_ssim(output.float(), target.float())

                    intensity_mae, chroma_mae, high_clip = (
                        calculate_hvi_metrics(
                            model,
                            output.float(),
                            target.float(),
                        )
                    )

                    if lpips_metric is not None:
                        lpips_value = calculate_lpips(
                            output.float(),
                            target.float(),
                            lpips_metric,
                            args.lpips_max_size,
                        )

                    pair_count += 1
                    sum_psnr += psnr
                    sum_ssim += ssim
                    sum_intensity += intensity_mae
                    sum_chroma += chroma_mae
                    sum_high_clip += high_clip

                    if lpips_value is not None:
                        sum_lpips += lpips_value
                        lpips_count += 1

                del target

            append_csv(
                csv_path,
                [
                    image_path.name,
                    str(gt_file) if gt_file is not None else "",
                    width,
                    height,
                    f"{elapsed:.6f}",
                    f"{psnr:.6f}" if psnr is not None else "",
                    f"{ssim:.6f}" if ssim is not None else "",
                    f"{lpips_value:.6f}" if lpips_value is not None else "",
                    f"{intensity_mae:.6f}" if intensity_mae is not None else "",
                    f"{chroma_mae:.6f}" if chroma_mae is not None else "",
                    f"{high_clip:.8f}",
                    f"{float(output.mean().cpu()):.8f}",
                    str(output_path),
                    status,
                ],
            )

            metric_text = (
                f"{width}x{height} | {elapsed:.3f}s | "
                f"clip={100.0 * high_clip:.3f}%"
            )

            if psnr is not None:
                metric_text += (
                    f" | PSNR={psnr:.4f} dB"
                    f" | SSIM={ssim:.6f}"
                    f" | I-MAE={intensity_mae:.6f}"
                    f" | Chroma-MAE={chroma_mae:.6f}"
                )

                if lpips_value is not None:
                    metric_text += f" | LPIPS={lpips_value:.6f}"

            print(metric_text, flush=True)

            del input_tensor, output

    mean_time = total_time / len(image_paths)

    summary = {
        "images": len(image_paths),
        "pairs": pair_count,
        "total_time": total_time,
        "mean_time": mean_time,
        "psnr": sum_psnr / pair_count if pair_count else 0.0,
        "ssim": sum_ssim / pair_count if pair_count else 0.0,
        "lpips": (
            sum_lpips / lpips_count
            if lpips_count
            else None
        ),
        "intensity_mae": (
            sum_intensity / pair_count
            if pair_count
            else 0.0
        ),
        "chroma_mae": (
            sum_chroma / pair_count
            if pair_count
            else 0.0
        ),
        "high_clip_fraction": (
            sum_high_clip / pair_count
            if pair_count
            else 0.0
        ),
    }

    write_summary(summary_path, summary)

    print("\n" + "=" * 84)
    print("INFERÊNCIA CONCLUÍDA")
    print(f"Tempo total:          {total_time:.3f}s")
    print(f"Tempo médio:          {mean_time:.3f}s/imagem")
    print(f"Pares avaliados:      {pair_count}")

    if pair_count:
        print(f"PSNR médio:           {summary['psnr']:.4f} dB")
        print(f"SSIM médio:           {summary['ssim']:.6f}")
        print(f"Intensity MAE médio:  {summary['intensity_mae']:.6f}")
        print(f"Chroma MAE médio:     {summary['chroma_mae']:.6f}")
        print(
            f"Pixels >= 0.999:      "
            f"{100.0 * summary['high_clip_fraction']:.3f}%"
        )

        if summary["lpips"] is not None:
            print(f"LPIPS médio:          {summary['lpips']:.6f}")

    print(f"Resultados:           {output_dir}")
    print(f"CSV:                  {csv_path}")
    print(f"Resumo:               {summary_path}")

    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Pico de memória:      {max_memory:.3f} GB")

    print("=" * 84)


if __name__ == "__main__":
    main()
