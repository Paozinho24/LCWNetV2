import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.lcw_hvi_backbone import LCWHVINet
from models.loss_hvi import LCWHVITotalLoss
from dataload.llie_dataset import DATASET_CHOICES, build_datasets_from_args

ARCHITECTURE_VERSION = "LCWHVINet_HVI_Restormer_v1"

LOSS_KEYS = (
    "loss_total", "loss_rgb", "loss_intensity", "loss_hv", "loss_chroma",
    "loss_hue", "loss_grad", "loss_curve_smooth", "loss_color_delta", "loss_ssim",
)


# Argumentos do treinamento.
def get_args():
    parser = argparse.ArgumentParser(
        description="Treinamento unificado da LCWHVINet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execução.
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--disable_ddp", action="store_true")

    # Dataset.
    parser.add_argument("--dataset_root", type=str, default="/home/unicornio/User/DataSetsLLIE")
    parser.add_argument("--dataset_name", type=str, default="lsd", choices=DATASET_CHOICES)
    parser.add_argument("--val_subset", type=str, default="DEI")
    parser.add_argument("--train_low_path", type=str, default="")
    parser.add_argument("--train_gt_path", type=str, default="")
    parser.add_argument("--val_low_path", type=str, default="")
    parser.add_argument("--val_gt_path", type=str, default="")
    parser.add_argument("--recursive_dataset", action="store_true")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument(
        "--patches_per_image",
        type=int,
        default=0,
        help=(
            "quantidade virtual de crops por imagem; "
            "0 usa automaticamente 1 para LSD/PAMAZONIA e 16 para LOL"
        ),
    )
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="mostra o progresso a cada N batches",
    )

    # Validação.
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_max_images", type=int, default=0)
    parser.add_argument("--val_max_side", type=int, default=1024)
    parser.add_argument("--val_num_workers", type=int, default=2)
    parser.add_argument("--disable_validation", action="store_true")

    # Arquitetura.
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--expansion_factor", type=float, default=2.66)
    parser.add_argument("--wavelet_mode", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--color_mode", type=str, default="lock", choices=["lock", "bounded"])
    parser.add_argument(
        "--color_unlock_epoch", type=int, default=-1,
        help="época em que lock muda para bounded; -1 mantém a cor travada",
    )
    parser.add_argument("--color_scale", type=float, default=0.03)
    parser.add_argument("--curve_steps", type=int, default=4)
    parser.add_argument("--curve_scale", type=float, default=1.0)
    parser.add_argument("--hvi_k", type=float, default=0.2)
    parser.add_argument(
        "--layernorm_type", type=str, default="WithBias",
        choices=["WithBias", "BiasFree"],
    )
    parser.add_argument("--bias", action="store_true")

    # Loss.
    parser.add_argument("--rgb_weight", type=float, default=1.0)
    parser.add_argument("--intensity_weight", type=float, default=0.5)
    parser.add_argument("--hv_weight", type=float, default=0.5)
    parser.add_argument("--chroma_weight", type=float, default=0.2)
    parser.add_argument("--hue_weight", type=float, default=0.1)
    parser.add_argument("--grad_weight", type=float, default=0.05)
    parser.add_argument("--curve_smooth_weight", type=float, default=0.02)
    parser.add_argument("--color_delta_weight", type=float, default=0.02)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--edge_strength", type=float, default=10.0)

    # Otimizador.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Checkpoints.
    parser.add_argument(
        "--checkpoints_dir", type=str,
        default="/home/unicornio/User/LCWNet/ckpt/LCWHVINet",
    )
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


# Valida os argumentos antes de iniciar o treino.
def validate_args(args):
    if args.channels <= 0:
        raise ValueError("channels precisa ser maior que zero.")
    if args.num_heads <= 0 or args.channels % args.num_heads != 0:
        raise ValueError("channels precisa ser divisível por num_heads.")
    if args.depth <= 0:
        raise ValueError("depth precisa ser maior que zero.")
    if args.patch_size <= 0:
        raise ValueError("patch_size precisa ser maior que zero.")
    if args.patches_per_image < 0:
        raise ValueError("patches_per_image não pode ser negativo.")
    if args.log_interval <= 0:
        raise ValueError("log_interval precisa ser maior que zero.")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size e epochs precisam ser maiores que zero.")
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("Use lr > 0 e 0 <= min_lr <= lr.")
    if args.val_every <= 0 or args.save_every <= 0:
        raise ValueError("val_every e save_every precisam ser maiores que zero.")
    if args.curve_steps <= 0 or args.curve_scale <= 0:
        raise ValueError("curve_steps e curve_scale precisam ser maiores que zero.")
    if args.color_scale < 0 or args.hvi_k <= 0:
        raise ValueError("color_scale >= 0 e hvi_k > 0 são obrigatórios.")
    if args.color_unlock_epoch == 0 or args.color_unlock_epoch < -1:
        raise ValueError("color_unlock_epoch deve ser -1 ou >= 1.")


# Inicializa o treinamento distribuído.
def setup_ddp():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# Define as seeds do treinamento.
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Warmup linear seguido de cosine decay.
def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs, base_lr, min_lr):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    min_factor = min_lr / base_lr

    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        cosine_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def get_model_module(model):
    return model.module if hasattr(model, "module") else model


# Controla a fase de cor durante o treinamento.
def runtime_color_mode(args, epoch):
    if args.color_mode == "bounded":
        return "bounded"
    if args.color_unlock_epoch > 0 and (epoch + 1) >= args.color_unlock_epoch:
        return "bounded"
    return "lock"


def apply_color_mode(model, criterion, mode):
    get_model_module(model).color_mode = mode
    criterion.set_color_mode(mode)


# Soma as estatísticas de todas as GPUs.
def reduce_train_statistics(
    metric_sums, sample_count, grad_norm_sum, batch_count,
    curve_abs_sum, delta_abs_sum, high_clip_sum, device,
):
    values = [metric_sums[key] for key in LOSS_KEYS]
    values += [
        sample_count, grad_norm_sum, batch_count,
        curve_abs_sum, delta_abs_sum, high_clip_sum,
    ]

    stats = torch.tensor(values, device=device, dtype=torch.float64)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    sample_total = max(1.0, stats[len(LOSS_KEYS)].item())
    batch_total = max(1.0, stats[len(LOSS_KEYS) + 2].item())

    averages = {
        key: stats[i].item() / sample_total
        for i, key in enumerate(LOSS_KEYS)
    }

    offset = len(LOSS_KEYS)
    averages["grad_norm"] = stats[offset + 1].item() / batch_total
    averages["curve_abs_mean"] = stats[offset + 3].item() / sample_total
    averages["delta_hv_abs_mean"] = stats[offset + 4].item() / sample_total
    averages["high_clip_fraction"] = stats[offset + 5].item() / sample_total
    return averages


# Executa uma época de treinamento.
def train_one_epoch(
    model, device, train_loader, optimizer, scheduler,
    criterion, epoch, rank, grad_clip, log_interval,
):
    model.train()
    total_batches = len(train_loader)
    log_interval = max(1, int(log_interval))

    metric_sums = {key: 0.0 for key in LOSS_KEYS}
    sample_count = 0
    batch_count = 0
    grad_norm_sum = 0.0
    curve_abs_sum = 0.0
    delta_abs_sum = 0.0
    high_clip_sum = 0.0

    epoch_start = time.time()
    end = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        data_time = time.time() - end

        # LOW e GT chegam em RGB [0,1].
        input_img = batch_data["input"].to(device, non_blocking=True)
        label = batch_data["label"].to(device, non_blocking=True)

        if input_img.min() < 0.0 or input_img.max() > 1.0:
            raise RuntimeError("Input fora de [0,1].")
        if label.min() < 0.0 or label.max() > 1.0:
            raise RuntimeError("GT fora de [0,1].")

        optimizer.zero_grad(set_to_none=True)

        # Forward com informações HVI auxiliares.
        output, aux = model(input_img, return_aux=True)
        loss, loss_logs = criterion(output, label, aux)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss inválida na época {epoch + 1}, batch {batch_idx + 1}: "
                f"{loss.item()}"
            )

        loss.backward()

        # Limita a norma global dos gradientes.
        if grad_clip > 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
            grad_norm = float(total_norm.detach().cpu())
        else:
            grad_norm = 0.0

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = input_img.size(0)
        sample_count += batch_size
        batch_count += 1
        grad_norm_sum += grad_norm

        for key in LOSS_KEYS:
            metric_sums[key] += float(loss_logs[key]) * batch_size

        # Diagnósticos para acompanhar iluminação e cor.
        curve_abs = float(aux["curve"].detach().abs().mean().cpu())
        delta_abs = float(aux["delta_hv"].detach().abs().mean().cpu())
        high_clip = float((output.detach() >= 0.999).float().mean().cpu())

        curve_abs_sum += curve_abs * batch_size
        delta_abs_sum += delta_abs * batch_size
        high_clip_sum += high_clip * batch_size

        batch_time = time.time() - end

        if rank == 0 and (
            batch_idx % log_interval == 0
            or batch_idx == total_batches - 1
        ):
            progress = 100.0 * (batch_idx + 1) / total_batches
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Batch {batch_idx + 1:04d}/{total_batches:04d} "
                f"({progress:5.1f}%) | "
                f"Loss {loss_logs['loss_total']:.6f} | "
                f"RGB {loss_logs['loss_rgb']:.6f} | "
                f"I {loss_logs['loss_intensity']:.6f} | "
                f"HV {loss_logs['loss_hv']:.6f} | "
                f"Chroma {loss_logs['loss_chroma']:.6f} | "
                f"Curve {curve_abs:.5f} | "
                f"DeltaHV {delta_abs:.5f} | "
                f"GradNorm {grad_norm:.4f} | "
                f"LR {lr:.3e} | "
                f"Data {data_time:.3f}s | "
                f"Batch {batch_time:.3f}s",
                flush=True,
            )

        end = time.time()

    averages = reduce_train_statistics(
        metric_sums, sample_count, grad_norm_sum, batch_count,
        curve_abs_sum, delta_abs_sum, high_clip_sum, device,
    )
    averages["epoch_seconds"] = time.time() - epoch_start
    averages["hvi_k"] = float(
        get_model_module(model).current_hvi_k().detach().cpu()
    )
    return averages


# Reduz imagens grandes apenas durante a validação.
def resize_validation_pair(input_img, label, max_side):
    if max_side <= 0:
        return input_img, label

    height, width = input_img.shape[-2:]
    largest = max(height, width)

    if largest <= max_side:
        return input_img, label

    scale = max_side / float(largest)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    input_img = F.interpolate(
        input_img, size=(new_h, new_w), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)

    label = F.interpolate(
        label, size=(new_h, new_w), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)

    return input_img, label


# Calcula PSNR em RGB [0,1].
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    return (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean()


def create_gaussian_kernel(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


# Calcula SSIM em RGB [0,1].
def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    _, channels, height, width = pred.shape
    size = min(window_size, height, width)

    if size % 2 == 0:
        size -= 1

    size = max(1, size)
    kernel = create_gaussian_kernel(
        size, sigma, channels, pred.device, pred.dtype
    )
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

    return (numerator / denominator.clamp_min(1e-12)).mean()


# Mede intensidade, cromaticidade e clipping separadamente.
def calculate_hvi_validation_metrics(model, pred, target):
    hvi_pred = model.hvi.rgb_to_hvi(pred)
    hvi_target = model.hvi.rgb_to_hvi(target)

    hv_pred = hvi_pred[:, 0:2]
    hv_target = hvi_target[:, 0:2]

    chroma_pred = torch.sqrt(
        hv_pred[:, 0:1].pow(2) + hv_pred[:, 1:2].pow(2) + 1e-8
    )
    chroma_target = torch.sqrt(
        hv_target[:, 0:1].pow(2) + hv_target[:, 1:2].pow(2) + 1e-8
    )

    intensity_mae = F.l1_loss(hvi_pred[:, 2:3], hvi_target[:, 2:3])
    chroma_mae = F.l1_loss(chroma_pred, chroma_target)
    high_clip = (pred >= 0.999).float().mean()

    return intensity_mae, chroma_mae, high_clip


# Valida a rede no rank 0.
def validate_one_epoch(
    model, device, val_loader, epoch,
    val_max_images=0, val_max_side=0,
):
    eval_model = get_model_module(model)
    was_training = eval_model.training
    eval_model.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    intensity_sum = 0.0
    chroma_sum = 0.0
    high_clip_sum = 0.0
    sample_count = 0
    start = time.time()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(val_loader):
            if val_max_images > 0 and sample_count >= val_max_images:
                break

            input_img = batch_data["input"].to(device, non_blocking=True)
            label = batch_data["label"].to(device, non_blocking=True)

            input_img, label = resize_validation_pair(
                input_img, label, val_max_side
            )

            output = eval_model(input_img).clamp(0.0, 1.0)

            psnr = calculate_psnr(output, label)
            ssim = calculate_ssim(output, label)
            intensity_mae, chroma_mae, high_clip = (
                calculate_hvi_validation_metrics(eval_model, output, label)
            )

            batch_size = input_img.size(0)
            psnr_sum += float(psnr.cpu()) * batch_size
            ssim_sum += float(ssim.cpu()) * batch_size
            intensity_sum += float(intensity_mae.cpu()) * batch_size
            chroma_sum += float(chroma_mae.cpu()) * batch_size
            high_clip_sum += float(high_clip.cpu()) * batch_size
            sample_count += batch_size

            scene = batch_data.get("scene_name", [f"imagem_{batch_idx}"])
            if isinstance(scene, (list, tuple)):
                scene = scene[0]

            print(
                f"[VALIDAÇÃO] Época {epoch + 1:03d} | {scene} | "
                f"PSNR {float(psnr):.4f} dB | "
                f"SSIM {float(ssim):.6f} | "
                f"I-MAE {float(intensity_mae):.6f} | "
                f"Chroma-MAE {float(chroma_mae):.6f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if was_training:
        eval_model.train()

    if sample_count == 0:
        raise RuntimeError("Nenhuma imagem processada na validação.")

    metrics = {
        "val_psnr": psnr_sum / sample_count,
        "val_ssim": ssim_sum / sample_count,
        "val_intensity_mae": intensity_sum / sample_count,
        "val_chroma_mae": chroma_sum / sample_count,
        "val_high_clip_fraction": high_clip_sum / sample_count,
        "val_images": sample_count,
        "val_seconds": time.time() - start,
    }

    print("\n" + "-" * 80)
    print(f"Validação época {epoch + 1}")
    print(f"PSNR:              {metrics['val_psnr']:.4f} dB")
    print(f"SSIM:              {metrics['val_ssim']:.6f}")
    print(f"Intensity MAE:     {metrics['val_intensity_mae']:.6f}")
    print(f"Chroma MAE:        {metrics['val_chroma_mae']:.6f}")
    print(f"Pixels >= 0.999:   {100.0 * metrics['val_high_clip_fraction']:.3f}%")
    print("-" * 80 + "\n", flush=True)

    return metrics


# Configuração que não pode mudar ao usar --resume.
def structural_config(args):
    return {
        "channels": args.channels,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "expansion_factor": args.expansion_factor,
        "wavelet_mode": args.wavelet_mode,
        "color_scale": args.color_scale,
        "curve_steps": args.curve_steps,
        "curve_scale": args.curve_scale,
        "hvi_k": args.hvi_k,
        "layernorm_type": args.layernorm_type,
        "bias": args.bias,
    }


# Salva checkpoint completo.
def save_checkpoint(
    path, epoch, model, optimizer, scheduler,
    metrics, args, active_color_mode,
):
    checkpoint = {
        "architecture_version": ARCHITECTURE_VERSION,
        "epoch": epoch,
        "model_state_dict": get_model_module(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": vars(args),
        "structural_config": structural_config(args),
        "active_color_mode": active_color_mode,
    }
    torch.save(checkpoint, path)


# Retoma um treinamento da mesma arquitetura.
def load_checkpoint(path, model, optimizer, scheduler, device, args):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    version = checkpoint.get("architecture_version", "")
    if version != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Checkpoint incompatível. Esperado={ARCHITECTURE_VERSION!r}; "
            f"encontrado={version or 'não registrado'!r}."
        )

    saved = checkpoint.get("structural_config", {})
    current = structural_config(args)

    for key, current_value in current.items():
        if key not in saved:
            continue

        saved_value = saved[key]

        if isinstance(current_value, float):
            equal = math.isclose(
                float(current_value), float(saved_value),
                rel_tol=1e-9, abs_tol=1e-12
            )
        else:
            equal = current_value == saved_value

        if not equal:
            raise ValueError(
                f"Configuração diferente em --resume: {key}. "
                f"Checkpoint={saved_value!r}; atual={current_value!r}."
            )

    get_model_module(model).load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"]) + 1, checkpoint.get("metrics", {})


# Cria o DataLoader de treinamento.
def create_train_loader(train_dataset, args, rank, world_size, use_ddp):
    if use_ddp and world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    generator = torch.Generator()
    generator.manual_seed(args.seed + rank)

    options = {
        "dataset": train_dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if args.num_workers > 0:
        options["prefetch_factor"] = 2

    return DataLoader(**options), sampler


def create_val_loader(val_dataset, args):
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.val_num_workers > 0,
    )


# Cria o cabeçalho do CSV.
def write_csv_header(path):
    fields = [
        "epoch", *LOSS_KEYS, "grad_norm", "curve_abs_mean",
        "delta_hv_abs_mean", "train_high_clip_fraction", "hvi_k",
        "color_mode", "lr", "epoch_seconds", "val_psnr", "val_ssim",
        "val_intensity_mae", "val_chroma_mae", "val_high_clip_fraction",
        "val_seconds", "best_psnr", "best_ssim", "best_chroma_mae",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fields)


# Adiciona uma época ao CSV.
def append_csv(
    path, epoch, train_metrics, val_metrics, color_mode,
    lr, best_psnr, best_ssim, best_chroma_mae,
):
    row = [
        epoch + 1,
        *[train_metrics[key] for key in LOSS_KEYS],
        train_metrics["grad_norm"],
        train_metrics["curve_abs_mean"],
        train_metrics["delta_hv_abs_mean"],
        train_metrics["high_clip_fraction"],
        train_metrics["hvi_k"],
        color_mode,
        lr,
        train_metrics["epoch_seconds"],
        val_metrics["val_psnr"],
        val_metrics["val_ssim"],
        val_metrics["val_intensity_mae"],
        val_metrics["val_chroma_mae"],
        val_metrics["val_high_clip_fraction"],
        val_metrics["val_seconds"],
        best_psnr,
        best_ssim,
        best_chroma_mae,
    ]

    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def main():
    args = get_args()

    # LSD/PAMAZONIA já são armazenados como patches prontos.
    # LOL usa imagens maiores e pode gerar vários crops virtuais por imagem.
    if args.patches_per_image == 0:
        if args.dataset_name in {"lsd", "pamazonia"}:
            args.patches_per_image = 1
        else:
            args.patches_per_image = 16

    validate_args(args)

    # Configurações de desempenho para A100/Ampere.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_ddp = (not args.disable_ddp) and ("RANK" in os.environ)

    try:
        # 1. GPU e DDP.
        if use_ddp:
            rank, local_rank, world_size = setup_ddp()
        else:
            rank, local_rank, world_size = 0, 0, 1
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

        set_random_seed(args.seed + rank)

        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if (
            rank == 0
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and world_size == 1
        ):
            print(
                "[AVISO] Mais de uma GPU está visível, mas o treinamento não está em DDP. "
                "Execute com torchrun --standalone --nproc_per_node=N.",
                flush=True,
            )

        # 2. Dataset unificado.
        train_dataset, val_dataset, dataset_paths = build_datasets_from_args(args)

        train_loader, train_sampler = create_train_loader(
            train_dataset, args, rank, world_size, use_ddp
        )

        if len(train_loader) == 0:
            raise RuntimeError("DataLoader vazio. Reduza batch_size.")

        val_loader = None

        if rank == 0 and not args.disable_validation:
            val_loader = create_val_loader(val_dataset, args)

        # 3. Modelo HVI + Wavelet + Restormer.
        initial_color_mode = runtime_color_mode(args, 0)

        model = LCWHVINet(
            channels=args.channels,
            num_heads=args.num_heads,
            depth=args.depth,
            expansion_factor=args.expansion_factor,
            wavelet_mode=args.wavelet_mode,
            color_mode=initial_color_mode,
            color_scale=args.color_scale,
            curve_steps=args.curve_steps,
            curve_scale=args.curve_scale,
            hvi_k=args.hvi_k,
            learnable_hvi_k=False,
            layernorm_type=args.layernorm_type,
            bias=args.bias,
        ).to(device)

        if use_ddp and world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        # 4. Loss HVI.
        criterion = LCWHVITotalLoss(
            rgb_weight=args.rgb_weight,
            intensity_weight=args.intensity_weight,
            hv_weight=args.hv_weight,
            chroma_weight=args.chroma_weight,
            hue_weight=args.hue_weight,
            grad_weight=args.grad_weight,
            curve_smooth_weight=args.curve_smooth_weight,
            color_delta_weight=args.color_delta_weight,
            ssim_weight=args.ssim_weight,
            color_mode=initial_color_mode,
            hvi_k=args.hvi_k,
            edge_strength=args.edge_strength,
        ).to(device)

        # 5. AdamW.
        trainable_params = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]

        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

        # 6. Scheduler.
        scheduler = build_scheduler(
            optimizer,
            args.epochs,
            len(train_loader),
            args.warmup_epochs,
            args.lr,
            args.min_lr,
        )

        # 7. Checkpoints e resume.
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        log_path = os.path.join(args.checkpoints_dir, "train_log.csv")

        start_epoch = 0
        best_psnr = float("-inf")
        best_ssim = float("-inf")
        best_chroma_mae = float("inf")

        if args.resume:
            start_epoch, old_metrics = load_checkpoint(
                args.resume, model, optimizer, scheduler, device, args
            )
            best_psnr = float(old_metrics.get("best_psnr", best_psnr))
            best_ssim = float(old_metrics.get("best_ssim", best_ssim))
            best_chroma_mae = float(
                old_metrics.get("best_chroma_mae", best_chroma_mae)
            )

        if rank == 0 and start_epoch == 0:
            write_csv_header(log_path)

        model_for_count = get_model_module(model)
        total_params = sum(p.numel() for p in model_for_count.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)

        # 8. Mostra a configuração.
        if rank == 0:
            print("\n" + "=" * 80)
            print("TREINAMENTO LCWHVINet — HVI + WAVELET + RESTORMER")
            print(f"Dataset:               {args.dataset_name}")
            print(f"Treino LOW:            {dataset_paths['train_low']}")
            print(f"Treino GT:             {dataset_paths['train_gt']}")
            print(f"Validação LOW:         {dataset_paths['val_low']}")
            print(f"Validação GT:          {dataset_paths['val_gt']}")
            print(f"Pares treino:          {len(train_dataset.pairs)}")
            print(f"Pares validação:       {len(val_dataset.pairs)}")
            print(f"Amostras/época:        {len(train_dataset)}")
            print(f"Patches por imagem:    {args.patches_per_image}")
            print(f"Patch:                 {args.patch_size}x{args.patch_size}")
            print(f"Batch/GPU:             {args.batch_size}")
            print(f"Batch global:          {args.batch_size * world_size}")
            print(f"GPUs/processos DDP:    {world_size}")
            print(f"GPUs CUDA visíveis:    {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            print(f"Log a cada:            {args.log_interval} batches")
            print(f"Channels:              {args.channels}")
            print(f"Depth:                 {args.depth}")
            print(f"Heads:                 {args.num_heads}")
            print(f"Wavelet:               {args.wavelet_mode}")
            print(f"Color mode:            {runtime_color_mode(args, start_epoch)}")
            print(
                f"Color unlock:          "
                f"{args.color_unlock_epoch if args.color_unlock_epoch > 0 else 'desativado'}"
            )
            print(f"Color scale:           {args.color_scale}")
            print(f"Curve steps:           {args.curve_steps}")
            print(f"HVI k:                 {args.hvi_k}")
            print(f"LR:                    {args.lr}")
            print(f"Grad clip:             {args.grad_clip}")
            print(f"Parâmetros totais:     {total_params / 1e6:.3f} M")
            print(f"Parâmetros treináveis: {trainable_count / 1e6:.3f} M")
            print(f"Device:                {device}")
            print("Pipeline:              RGB [0,1]")
            print("=" * 80 + "\n", flush=True)

        # 9. Loop de épocas.
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            active_color_mode = runtime_color_mode(args, epoch)
            apply_color_mode(model, criterion, active_color_mode)

            if rank == 0:
                print(
                    f"\n[EPOCH {epoch + 1}/{args.epochs}] "
                    f"color_mode={active_color_mode}",
                    flush=True,
                )

            train_metrics = train_one_epoch(
                model, device, train_loader, optimizer, scheduler,
                criterion, epoch, rank, args.grad_clip, args.log_interval,
            )

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if use_ddp and world_size > 1:
                dist.barrier()

            val_metrics = {
                "val_psnr": float("nan"),
                "val_ssim": float("nan"),
                "val_intensity_mae": float("nan"),
                "val_chroma_mae": float("nan"),
                "val_high_clip_fraction": float("nan"),
                "val_images": 0,
                "val_seconds": 0.0,
            }

            should_validate = (
                not args.disable_validation
                and (epoch + 1) % args.val_every == 0
            )

            if rank == 0 and should_validate:
                val_metrics = validate_one_epoch(
                    model, device, val_loader, epoch,
                    args.val_max_images, args.val_max_side,
                )

            if use_ddp and world_size > 1:
                dist.barrier()

            # 10. Salva logs e checkpoints no rank 0.
            if rank == 0:
                improved_psnr = False
                improved_ssim = False
                improved_chroma = False

                if should_validate:
                    if val_metrics["val_psnr"] > best_psnr:
                        best_psnr = val_metrics["val_psnr"]
                        improved_psnr = True

                    if val_metrics["val_ssim"] > best_ssim:
                        best_ssim = val_metrics["val_ssim"]
                        improved_ssim = True

                    if val_metrics["val_chroma_mae"] < best_chroma_mae:
                        best_chroma_mae = val_metrics["val_chroma_mae"]
                        improved_chroma = True

                checkpoint_metrics = {
                    **train_metrics,
                    **val_metrics,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_chroma_mae": best_chroma_mae,
                }

                lr = optimizer.param_groups[0]["lr"]

                append_csv(
                    log_path, epoch, train_metrics, val_metrics,
                    active_color_mode, lr,
                    best_psnr, best_ssim, best_chroma_mae,
                )

                latest_path = os.path.join(args.checkpoints_dir, "latest.pth")

                save_checkpoint(
                    latest_path, epoch, model, optimizer, scheduler,
                    checkpoint_metrics, args, active_color_mode,
                )

                if improved_psnr:
                    save_checkpoint(
                        os.path.join(args.checkpoints_dir, "best_psnr.pth"),
                        epoch, model, optimizer, scheduler,
                        checkpoint_metrics, args, active_color_mode,
                    )

                if improved_ssim:
                    save_checkpoint(
                        os.path.join(args.checkpoints_dir, "best_ssim.pth"),
                        epoch, model, optimizer, scheduler,
                        checkpoint_metrics, args, active_color_mode,
                    )

                # Guarda também o checkpoint com menor erro cromático.
                if improved_chroma:
                    save_checkpoint(
                        os.path.join(args.checkpoints_dir, "best_chroma.pth"),
                        epoch, model, optimizer, scheduler,
                        checkpoint_metrics, args, active_color_mode,
                    )

                if (
                    (epoch + 1) % args.save_every == 0
                    or epoch == args.epochs - 1
                ):
                    save_checkpoint(
                        os.path.join(
                            args.checkpoints_dir,
                            f"epoch_{epoch + 1:04d}.pth",
                        ),
                        epoch, model, optimizer, scheduler,
                        checkpoint_metrics, args, active_color_mode,
                    )

                print("\n" + "=" * 80)
                print(f"RESUMO ÉPOCA {epoch + 1}")
                print(f"Color mode:         {active_color_mode}")
                print(f"Loss total:         {train_metrics['loss_total']:.6f}")
                print(f"Loss RGB:           {train_metrics['loss_rgb']:.6f}")
                print(f"Loss intensidade:   {train_metrics['loss_intensity']:.6f}")
                print(f"Loss HV:            {train_metrics['loss_hv']:.6f}")
                print(f"Loss chroma:        {train_metrics['loss_chroma']:.6f}")
                print(f"Curve |mean|:       {train_metrics['curve_abs_mean']:.6f}")
                print(f"Delta HV |mean|:    {train_metrics['delta_hv_abs_mean']:.6f}")
                print(f"Grad norm:          {train_metrics['grad_norm']:.6f}")
                print(
                    f"Pixels treino >=1:  "
                    f"{100.0 * train_metrics['high_clip_fraction']:.3f}%"
                )

                if should_validate:
                    print(f"Val PSNR:           {val_metrics['val_psnr']:.4f} dB")
                    print(f"Val SSIM:           {val_metrics['val_ssim']:.6f}")
                    print(f"Val I-MAE:          {val_metrics['val_intensity_mae']:.6f}")
                    print(f"Val Chroma-MAE:     {val_metrics['val_chroma_mae']:.6f}")
                    print(
                        f"Val pixels >=1:     "
                        f"{100.0 * val_metrics['val_high_clip_fraction']:.3f}%"
                    )

                print(f"Duração:            {train_metrics['epoch_seconds']:.2f}s")
                print(f"Checkpoint:         {latest_path}")
                print("=" * 80 + "\n", flush=True)

            if use_ddp and world_size > 1:
                dist.barrier()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()