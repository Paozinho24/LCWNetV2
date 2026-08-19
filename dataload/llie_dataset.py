from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

# Extensões aceitas pelos datasets LLIE.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# Nomes aceitos pelo script único de treinamento/inferência.
DATASET_CHOICES = (
    "lsd",
    "pamazonia",
    "lolv1",
    "lolv2_real",
    "lolv2_synthetic",
)


def _is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _list_images(directory, recursive=False):
    """Lista imagens de uma pasta em ordem estável."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {directory}")

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    images = sorted(
        [path for path in iterator if _is_image(path)],
        key=lambda path: str(path).lower(),
    )

    if not images:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {directory}")

    return images


def _find_child_case_insensitive(parent, candidates):
    """Localiza uma subpasta ignorando diferenças entre maiúsculas/minúsculas."""
    parent = Path(parent)

    if not parent.is_dir():
        return None

    children = {
        child.name.lower(): child
        for child in parent.iterdir()
        if child.is_dir()
    }

    for candidate in candidates:
        match = children.get(candidate.lower())
        if match is not None:
            return match

    return None


def _require_child(parent, candidates, description):
    """Localiza uma subpasta e mostra um erro claro quando ela não existe."""
    result = _find_child_case_insensitive(parent, candidates)

    if result is None:
        expected = ", ".join(candidates)
        raise FileNotFoundError(
            f"Não foi possível localizar {description} dentro de {parent}. "
            f"Nomes procurados: {expected}"
        )

    return result


def _resolve_dataset_base(dataset_root, candidates):
    """
    Localiza a pasta principal de um dataset.

    Também aceita dataset_root apontando diretamente para a pasta do dataset.
    """
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"dataset_root não encontrado: {root}")

    if root.is_file():
        raise ValueError(f"dataset_root precisa ser uma pasta: {root}")

    if root.name.lower() in {name.lower() for name in candidates}:
        return root

    result = _find_child_case_insensitive(root, candidates)

    if result is None:
        expected = ", ".join(candidates)
        raise FileNotFoundError(
            f"Dataset não encontrado dentro de {root}. "
            f"Pastas procuradas: {expected}"
        )

    return result


def resolve_dataset_paths(
    dataset_root,
    dataset_name,
    val_subset="DEI",
    train_low_path="",
    train_gt_path="",
    val_low_path="",
    val_gt_path="",
):
    """
    Resolve automaticamente as pastas LOW/GT de treinamento e validação.

    Caminhos informados explicitamente têm prioridade sobre os caminhos
    automáticos. Isso permite usar o mesmo arquivo de treino para todos
    os datasets apenas alterando parser.add_argument.
    """
    dataset_name = str(dataset_name).lower().strip()

    if dataset_name not in DATASET_CHOICES:
        raise ValueError(
            f"dataset_name inválido: {dataset_name!r}. "
            f"Opções: {', '.join(DATASET_CHOICES)}"
        )

    explicit = {
        "train_low": Path(train_low_path) if train_low_path else None,
        "train_gt": Path(train_gt_path) if train_gt_path else None,
        "val_low": Path(val_low_path) if val_low_path else None,
        "val_gt": Path(val_gt_path) if val_gt_path else None,
    }

    # LSD/PAMAZONIA: usa inputPatchDLL/gtPatchDLL no treino e
    # Testing/In-the-wild/SUBSET/SUBSET_LOW|GT na validação.
    if dataset_name in {"lsd", "pamazonia"}:
        base_candidates = (
            ("LSD", "lsd")
            if dataset_name == "lsd"
            else ("PAMAZONIA", "PamAmazonia", "pamazonia", "LSD", "lsd")
        )

        base = _resolve_dataset_base(dataset_root, base_candidates)

        train_low = explicit["train_low"] or _require_child(
            base,
            ("inputPatchDLL", "inputpatchdll"),
            "a pasta LOW de treinamento",
        )
        train_gt = explicit["train_gt"] or _require_child(
            base,
            ("gtPatchDLL", "gtpatchdll"),
            "a pasta GT de treinamento",
        )

        testing = base / "Testing" / "In-the-wild"
        subset = testing / val_subset

        val_low = explicit["val_low"] or (
            subset / f"{val_subset}_LOW"
        )
        val_gt = explicit["val_gt"] or (
            subset / f"{val_subset}_GT"
        )

    # LOL-v1:
    #   our485/low  -> treino
    #   our485/high -> GT treino
    #   eval15/low  -> validação
    #   eval15/high -> GT validação
    elif dataset_name == "lolv1":
        base = _resolve_dataset_base(
            dataset_root,
            ("LOLv1", "LOL-v1", "LOL_V1", "LOL"),
        )

        train_root = _require_child(
            base,
            ("our485",),
            "o conjunto de treinamento our485",
        )
        val_root = _require_child(
            base,
            ("eval15",),
            "o conjunto de validação eval15",
        )

        train_low = explicit["train_low"] or _require_child(
            train_root,
            ("low", "Low"),
            "a pasta LOW de treinamento",
        )
        train_gt = explicit["train_gt"] or _require_child(
            train_root,
            ("high", "High", "normal", "Normal"),
            "a pasta GT de treinamento",
        )
        val_low = explicit["val_low"] or _require_child(
            val_root,
            ("low", "Low"),
            "a pasta LOW de validação",
        )
        val_gt = explicit["val_gt"] or _require_child(
            val_root,
            ("high", "High", "normal", "Normal"),
            "a pasta GT de validação",
        )

    # LOL-v2:
    #   Real_captured/Train/Low|Normal
    #   Real_captured/Test/Low|Normal
    # ou a mesma estrutura em Synthetic.
    else:
        base = _resolve_dataset_base(
            dataset_root,
            ("LOLv2", "LOL-v2", "LOL_V2"),
        )

        subset_candidates = (
            ("Real_captured", "RealCaptured", "real_captured", "real")
            if dataset_name == "lolv2_real"
            else ("Synthetic", "synthetic")
        )

        subset_root = _require_child(
            base,
            subset_candidates,
            "o subconjunto do LOL-v2",
        )

        train_root = _require_child(
            subset_root,
            ("Train", "train"),
            "a pasta Train",
        )
        val_root = _require_child(
            subset_root,
            ("Test", "test"),
            "a pasta Test",
        )

        train_low = explicit["train_low"] or _require_child(
            train_root,
            ("Low", "low"),
            "a pasta LOW de treinamento",
        )
        train_gt = explicit["train_gt"] or _require_child(
            train_root,
            ("Normal", "normal", "High", "high"),
            "a pasta GT de treinamento",
        )
        val_low = explicit["val_low"] or _require_child(
            val_root,
            ("Low", "low"),
            "a pasta LOW de validação",
        )
        val_gt = explicit["val_gt"] or _require_child(
            val_root,
            ("Normal", "normal", "High", "high"),
            "a pasta GT de validação",
        )

    paths = {
        "dataset_name": dataset_name,
        "train_low": Path(train_low),
        "train_gt": Path(train_gt),
        "val_low": Path(val_low),
        "val_gt": Path(val_gt),
    }

    # Valida todos os caminhos resolvidos antes de iniciar o treinamento.
    for name, path in paths.items():
        if name == "dataset_name":
            continue
        if not path.is_dir():
            raise FileNotFoundError(
                f"Pasta {name} não encontrada: {path}. "
                "Use os argumentos de caminho explícito caso sua estrutura "
                "de diretórios seja diferente."
            )

    return paths


def _candidate_pair_stems(stem):
    """Gera nomes possíveis para localizar o par LOW/GT."""
    candidates = [stem]
    lower = stem.lower()

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
        if lower.endswith(source):
            candidates.append(stem[:-len(source)] + target)
            candidates.append(stem[:-len(source)])

    # Também permite localizar LOW a partir de nomes GT/high/normal.
    reverse_replacements = (
        ("_gt", "_low"),
        ("-gt", "-low"),
        ("_high", "_low"),
        ("-high", "-low"),
        ("_normal", "_low"),
        ("-normal", "-low"),
    )

    for source, target in reverse_replacements:
        if lower.endswith(source):
            candidates.append(stem[:-len(source)] + target)
            candidates.append(stem[:-len(source)])

    unique = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in {item.lower() for item in unique}:
            unique.append(candidate)

    return unique


def _build_image_map(directory, recursive=False):
    """Cria mapa por nome-base para pareamento rápido."""
    images = _list_images(directory, recursive=recursive)

    image_map = {}

    for path in images:
        key = path.stem.lower()

        if key in image_map:
            raise RuntimeError(
                f"Nome-base duplicado em {directory}: {path.stem}. "
                "Use uma estrutura sem nomes repetidos ou desative recursive."
            )

        image_map[key] = path

    return images, image_map


def build_pairs(low_dir, gt_dir, recursive=False):
    """
    Cria pares LOW/GT.

    Primeiro tenta o mesmo nome-base. Depois tenta padrões como
    *_LOW -> *_GT e *_LOW -> *_HIGH.
    """
    low_images = _list_images(low_dir, recursive=recursive)
    _, gt_map = _build_image_map(gt_dir, recursive=recursive)

    pairs = []
    missing = []

    for low_path in low_images:
        gt_path = None

        for candidate in _candidate_pair_stems(low_path.stem):
            match = gt_map.get(candidate.lower())
            if match is not None:
                gt_path = match
                break

        if gt_path is None:
            missing.append(low_path)
        else:
            pairs.append((low_path, gt_path))

    if not pairs:
        raise RuntimeError(
            f"Nenhum par LOW/GT válido encontrado.\n"
            f"LOW: {low_dir}\n"
            f"GT:  {gt_dir}"
        )

    return pairs, missing


def _load_rgb_tensor(path):
    """
    Carrega a imagem explicitamente como RGB e retorna float32 em [0,1].

    Não existe conversão para [-1,1] e não existe BGR neste pipeline.
    """
    with Image.open(path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.clamp(0.0, 1.0)


def _resize_pair_if_needed(low, gt, minimum_size):
    """
    Amplia LOW e GT juntos apenas quando são menores que o patch solicitado.
    Mantém o alinhamento espacial entre as duas imagens.
    """
    if minimum_size <= 0:
        return low, gt

    height, width = low.shape[-2:]

    if height >= minimum_size and width >= minimum_size:
        return low, gt

    scale = max(
        minimum_size / float(height),
        minimum_size / float(width),
    )

    new_h = max(minimum_size, int(round(height * scale)))
    new_w = max(minimum_size, int(round(width * scale)))

    low = F.interpolate(
        low.unsqueeze(0),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    gt = F.interpolate(
        gt.unsqueeze(0),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    return low.clamp(0.0, 1.0), gt.clamp(0.0, 1.0)


def _random_crop_pair(low, gt, patch_size):
    """Aplica exatamente o mesmo crop em LOW e GT."""
    if patch_size <= 0:
        return low, gt

    height, width = low.shape[-2:]

    if height < patch_size or width < patch_size:
        raise RuntimeError(
            f"Imagem menor que patch_size após preparação: "
            f"{height}x{width}, patch={patch_size}"
        )

    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)

    low = low[:, top:top + patch_size, left:left + patch_size]
    gt = gt[:, top:top + patch_size, left:left + patch_size]

    return low, gt


def _augment_pair(low, gt):
    """
    Augmentação geométrica sincronizada.

    Não usa color jitter, gamma ou qualquer transformação cromática,
    evitando contaminar a relação de cor entre LOW e GT.
    """
    if random.random() < 0.5:
        low = torch.flip(low, dims=(2,))
        gt = torch.flip(gt, dims=(2,))

    if random.random() < 0.5:
        low = torch.flip(low, dims=(1,))
        gt = torch.flip(gt, dims=(1,))

    rotation = random.randint(0, 3)

    if rotation:
        low = torch.rot90(low, rotation, dims=(1, 2))
        gt = torch.rot90(gt, rotation, dims=(1, 2))

    return low.contiguous(), gt.contiguous()


class LLIETrainDataset(Dataset):
    """
    Dataset de treinamento unificado para LSD/PAMAZONIA e LOL.

    Todas as imagens são retornadas como RGB float32 no intervalo [0,1].
    """

    def __init__(
        self,
        low_dir,
        gt_dir,
        dataset_name,
        patch_size=128,
        patches_per_image=16,
        augment=True,
        recursive=False,
    ):
        super().__init__()

        self.low_dir = Path(low_dir)
        self.gt_dir = Path(gt_dir)
        self.dataset_name = str(dataset_name)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = bool(augment)
        self.recursive = bool(recursive)

        if self.patch_size <= 0:
            raise ValueError("patch_size precisa ser maior que zero.")

        if self.patches_per_image <= 0:
            raise ValueError("patches_per_image precisa ser maior que zero.")

        self.pairs, self.missing_pairs = build_pairs(
            self.low_dir,
            self.gt_dir,
            recursive=self.recursive,
        )

    def __len__(self):
        # Repete virtualmente cada par para produzir vários crops por época.
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index):
        pair_index = index % len(self.pairs)
        low_path, gt_path = self.pairs[pair_index]

        low = _load_rgb_tensor(low_path)
        gt = _load_rgb_tensor(gt_path)

        if low.shape != gt.shape:
            raise RuntimeError(
                "LOW e GT precisam estar perfeitamente alinhados.\n"
                f"LOW: {low_path} -> {tuple(low.shape)}\n"
                f"GT:  {gt_path} -> {tuple(gt.shape)}"
            )

        # Garante espaço suficiente para o crop sem deformar apenas um dos pares.
        low, gt = _resize_pair_if_needed(
            low,
            gt,
            self.patch_size,
        )

        # Usa o mesmo crop nas duas imagens.
        low, gt = _random_crop_pair(
            low,
            gt,
            self.patch_size,
        )

        # Aplica somente augmentações geométricas sincronizadas.
        if self.augment:
            low, gt = _augment_pair(low, gt)

        return {
            "input": low,
            "label": gt,
            "scene_name": low_path.stem,
            "low_path": str(low_path),
            "gt_path": str(gt_path),
            "dataset_name": self.dataset_name,
        }


class LLIEValDataset(Dataset):
    """
    Dataset de validação unificado.

    A validação usa a imagem completa, sem crop e sem augmentação.
    """

    def __init__(
        self,
        low_dir,
        gt_dir,
        dataset_name,
        recursive=False,
    ):
        super().__init__()

        self.low_dir = Path(low_dir)
        self.gt_dir = Path(gt_dir)
        self.dataset_name = str(dataset_name)
        self.recursive = bool(recursive)

        self.pairs, self.missing_pairs = build_pairs(
            self.low_dir,
            self.gt_dir,
            recursive=self.recursive,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        low_path, gt_path = self.pairs[index]

        low = _load_rgb_tensor(low_path)
        gt = _load_rgb_tensor(gt_path)

        if low.shape != gt.shape:
            raise RuntimeError(
                "LOW e GT precisam ter as mesmas dimensões na validação.\n"
                f"LOW: {low_path} -> {tuple(low.shape)}\n"
                f"GT:  {gt_path} -> {tuple(gt.shape)}"
            )

        return {
            "input": low,
            "label": gt,
            "scene_name": low_path.stem,
            "low_path": str(low_path),
            "gt_path": str(gt_path),
            "dataset_name": self.dataset_name,
        }


def build_train_dataset(
    dataset_root,
    dataset_name,
    patch_size=128,
    patches_per_image=16,
    augment=True,
    recursive=False,
    val_subset="DEI",
    train_low_path="",
    train_gt_path="",
    val_low_path="",
    val_gt_path="",
):
    """Cria o dataset de treinamento a partir do nome do dataset."""
    paths = resolve_dataset_paths(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        val_subset=val_subset,
        train_low_path=train_low_path,
        train_gt_path=train_gt_path,
        val_low_path=val_low_path,
        val_gt_path=val_gt_path,
    )

    return LLIETrainDataset(
        low_dir=paths["train_low"],
        gt_dir=paths["train_gt"],
        dataset_name=dataset_name,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        augment=augment,
        recursive=recursive,
    )


def build_val_dataset(
    dataset_root,
    dataset_name,
    recursive=False,
    val_subset="DEI",
    train_low_path="",
    train_gt_path="",
    val_low_path="",
    val_gt_path="",
):
    """Cria o dataset de validação a partir do nome do dataset."""
    paths = resolve_dataset_paths(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        val_subset=val_subset,
        train_low_path=train_low_path,
        train_gt_path=train_gt_path,
        val_low_path=val_low_path,
        val_gt_path=val_gt_path,
    )

    return LLIEValDataset(
        low_dir=paths["val_low"],
        gt_dir=paths["val_gt"],
        dataset_name=dataset_name,
        recursive=recursive,
    )


def build_datasets_from_args(args):
    """
    Cria treino e validação diretamente a partir do argparse.Namespace.

    O futuro train_lcw_hvi.py poderá usar apenas:
        train_dataset, val_dataset, paths = build_datasets_from_args(args)
    """
    paths = resolve_dataset_paths(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        val_subset=getattr(args, "val_subset", "DEI"),
        train_low_path=getattr(args, "train_low_path", ""),
        train_gt_path=getattr(args, "train_gt_path", ""),
        val_low_path=getattr(args, "val_low_path", ""),
        val_gt_path=getattr(args, "val_gt_path", ""),
    )

    train_dataset = LLIETrainDataset(
        low_dir=paths["train_low"],
        gt_dir=paths["train_gt"],
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        augment=not getattr(args, "disable_augmentation", False),
        recursive=getattr(args, "recursive_dataset", False),
    )

    val_dataset = LLIEValDataset(
        low_dir=paths["val_low"],
        gt_dir=paths["val_gt"],
        dataset_name=args.dataset_name,
        recursive=getattr(args, "recursive_dataset", False),
    )

    return train_dataset, val_dataset, paths
