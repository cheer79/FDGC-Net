

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Utils.fdgc_model import FDGCCIFARNet, fdgc_total_loss

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="FDGC objective for CIFAR-10 / CIFAR-100")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./Weight")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU index to use when --device starts with cuda")
    parser.add_argument("--strict_cuda", action="store_true", default=True, help="Error out if CUDA is unavailable")
    parser.add_argument("--allow_cpu_fallback", action="store_false", dest="strict_cuda")
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "simple"])
    parser.add_argument("--auto_augment", action="store_true", default=True)
    parser.add_argument("--no_auto_augment", action="store_false", dest="auto_augment")
    parser.add_argument("--random_erasing", type=float, default=0.25)
    parser.add_argument("--fdgc_warmup_epochs", type=int, default=40)

    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--dictionary_size", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--transform_hidden", type=int, default=64)

    parser.add_argument("--gamma_a", type=float, default=0.02)
    parser.add_argument("--gamma_s", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=0.005)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--topk_atom", type=int, default=16)
    parser.add_argument("--topk_token", type=int, default=8)
    parser.add_argument("--sigma_atom", type=float, default=1.0)
    parser.add_argument("--sigma_token", type=float, default=1.0)

    parser.add_argument("--fake_data", action="store_true", help="Use torchvision FakeData for offline smoke tests")
    parser.add_argument("--fake_train_size", type=int, default=2048)
    parser.add_argument("--fake_test_size", type=int, default=512)
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no_download", action="store_false", dest="download")

    parser.add_argument("--curve_font_family", type=str, default="DejaVu Sans")
    parser.add_argument("--curve_title_fontsize", type=float, default=14.0)
    parser.add_argument("--curve_axis_label_fontsize", type=float, default=12.0)
    parser.add_argument("--curve_tick_fontsize", type=float, default=10.0)
    parser.add_argument("--curve_legend_fontsize", type=float, default=10.0)
    parser.add_argument("--curve_legend_loc", type=str, default="upper right")
    parser.add_argument("--curve_legend_bbox", type=str, default="", help="Legend bbox anchor for curve figure, format x,y")
    parser.add_argument("--curve_legend_framealpha", type=float, default=0.75)
    parser.add_argument("--curve_shadow_alpha", type=float, default=0.18)
    parser.add_argument("--curve_shadow_scale", type=float, default=1.0)
    parser.add_argument("--curve_shadow_window", type=int, default=7)

    parser.add_argument("--embed_font_family", type=str, default="DejaVu Sans")
    parser.add_argument("--embed_title_fontsize", type=float, default=14.0)
    parser.add_argument("--embed_axis_label_fontsize", type=float, default=12.0)
    parser.add_argument("--embed_tick_fontsize", type=float, default=10.0)
    parser.add_argument("--embed_legend_fontsize", type=float, default=9.0)
    parser.add_argument("--embed_legend_loc", type=str, default="best")
    parser.add_argument("--embed_legend_bbox", type=str, default="", help="Legend bbox anchor for embedding figure, format x,y")
    parser.add_argument("--embed_legend_framealpha", type=float, default=0.75)
    parser.add_argument("--embed_point_alpha", type=float, default=0.85)
    parser.add_argument("--embed_point_size", type=float, default=16.0)
    parser.add_argument("--embed_max_points", type=int, default=3000)
    parser.add_argument("--embed_perplexity", type=int, default=30)
    parser.add_argument("--embed_max_legend_classes", type=int, default=20)
    parser.add_argument("--save_embedding_vis", action="store_true", default=True)
    parser.add_argument("--no_save_embedding_vis", action="store_false", dest="save_embedding_vis")

    parser.add_argument("--gradcam_num_images", type=int, default=4)
    parser.add_argument("--gradcam_cols", type=int, default=4)
    parser.add_argument("--gradcam_alpha", type=float, default=0.45)
    parser.add_argument("--gradcam_title_fontsize", type=float, default=9.0)
    parser.add_argument("--gradcam_suptitle_fontsize", type=float, default=14.0)
    parser.add_argument("--save_gradcam_vis", action="store_true", default=True)
    parser.add_argument("--no_save_gradcam_vis", action="store_false", dest="save_gradcam_vis")
    return parser.parse_args()


def build_run_paths(args):
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.dataset}-{run_time}"
    run_dir = os.path.join(args.save_dir, run_name)
    if os.path.exists(run_dir):
        suffix = 1
        while True:
            alt_name = f"{run_name}-{suffix:02d}"
            alt_dir = os.path.join(args.save_dir, alt_name)
            if not os.path.exists(alt_dir):
                run_name = alt_name
                run_dir = alt_dir
                break
            suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "best_path": os.path.join(run_dir, "best.pth"),
        "last_path": os.path.join(run_dir, "last.pth"),
        "log_path": os.path.join(run_dir, f"{run_name}.log"),
        "metric_csv_path": os.path.join(run_dir, f"{run_name}_metrics.csv"),
        "curve_path": os.path.join(run_dir, f"{run_name}_convergence.png"),
        "embedding_path": os.path.join(run_dir, f"{run_name}_embedding2d.png"),
        "gradcam_path": os.path.join(run_dir, f"{run_name}_gradcam.png"),
    }


def resolve_save_dir(save_dir: str) -> str:
    script_root = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.path.isabs(save_dir):
        candidates.append(save_dir)
    else:
        # 1) respect current working directory first
        candidates.append(os.path.join(os.getcwd(), save_dir))
        # 2) fallback to script directory
        candidates.append(os.path.join(script_root, save_dir))
    # final fallback: project Weight folder
    candidates.append(os.path.join(script_root, "Weight"))

    checked = set()
    last_err = None
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in checked:
            continue
        checked.add(abs_path)
        try:
            os.makedirs(abs_path, exist_ok=True)
            probe_path = os.path.join(abs_path, f".write_probe_{os.getpid()}.tmp")
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe_path)
            return abs_path
        except OSError as e:
            last_err = e

    raise PermissionError(
        f"Cannot create writable save_dir from '{save_dir}'. "
        f"Tried: {', '.join(sorted(checked))}. Last error: {last_err}"
    )


def save_history_csv(history: Dict[str, list], csv_path: str):
    if not history["epoch"]:
        return
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def parse_legend_bbox(legend_bbox: str) -> Optional[Tuple[float, float]]:
    text = (legend_bbox or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid legend bbox '{legend_bbox}', expected format x,y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid legend bbox '{legend_bbox}', expected numeric x,y") from exc


def _clamp_01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _smooth_and_std(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values, np.zeros_like(values)
    window = max(int(window), 1)
    if window <= 1 or values.size <= 2:
        return values, np.zeros_like(values)

    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left

    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    mean = np.convolve(padded, kernel, mode="valid")

    padded2 = np.pad(values * values, (pad_left, pad_right), mode="edge")
    mean2 = np.convolve(padded2, kernel, mode="valid")
    var = np.clip(mean2 - mean * mean, a_min=0.0, a_max=None)
    std = np.sqrt(var)
    return mean, std


def _style_axis_ticks(ax, font_family: str, tick_size: float):
    ax.tick_params(labelsize=tick_size)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font_family)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font_family)


def _plot_with_shadow(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    shadow_window: int,
    shadow_scale: float,
    shadow_alpha: float,
):
    y_smooth, y_std = _smooth_and_std(y, window=shadow_window)
    band = y_std * max(float(shadow_scale), 0.0)
    (line,) = ax.plot(x, y_smooth, color=color, linewidth=2.0, label=label)
    if np.any(band > 0):
        ax.fill_between(x, y_smooth - band, y_smooth + band, color=color, alpha=shadow_alpha, linewidth=0.0)
    return line


def save_convergence_curve(history: Dict[str, list], figure_path: str, args):
    if not history["epoch"]:
        return

    epochs = np.asarray(history["epoch"], dtype=np.int32)
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    test_loss = np.asarray(history["test_loss"], dtype=np.float64)
    train_acc = np.asarray(history["train_clean_acc"], dtype=np.float64)
    test_acc = np.asarray(history["test_acc"], dtype=np.float64)

    fig, ax_loss = plt.subplots(figsize=(9.8, 5.6))
    ax_acc = ax_loss.twinx()

    shadow_alpha = _clamp_01(args.curve_shadow_alpha)
    framealpha = _clamp_01(args.curve_legend_framealpha)
    legend_bbox = parse_legend_bbox(args.curve_legend_bbox)

    line_train_loss = _plot_with_shadow(
        ax=ax_loss,
        x=epochs,
        y=train_loss,
        color="#d62728",
        label="train_loss",
        shadow_window=args.curve_shadow_window,
        shadow_scale=args.curve_shadow_scale,
        shadow_alpha=shadow_alpha,
    )
    line_test_loss = _plot_with_shadow(
        ax=ax_loss,
        x=epochs,
        y=test_loss,
        color="#ff7f0e",
        label="test_loss",
        shadow_window=args.curve_shadow_window,
        shadow_scale=args.curve_shadow_scale,
        shadow_alpha=shadow_alpha,
    )
    line_train_acc = _plot_with_shadow(
        ax=ax_acc,
        x=epochs,
        y=train_acc,
        color="#1f77b4",
        label="train_acc",
        shadow_window=args.curve_shadow_window,
        shadow_scale=args.curve_shadow_scale,
        shadow_alpha=shadow_alpha,
    )
    line_test_acc = _plot_with_shadow(
        ax=ax_acc,
        x=epochs,
        y=test_acc,
        color="#2ca02c",
        label="test_acc",
        shadow_window=args.curve_shadow_window,
        shadow_scale=args.curve_shadow_scale,
        shadow_alpha=shadow_alpha,
    )

    ax_loss.set_title(
        "Training Curves (Loss + Acc)",
        fontfamily=args.curve_font_family,
        fontsize=args.curve_title_fontsize,
    )
    ax_loss.set_xlabel("Epoch", fontfamily=args.curve_font_family, fontsize=args.curve_axis_label_fontsize)
    ax_loss.set_ylabel("Loss", fontfamily=args.curve_font_family, fontsize=args.curve_axis_label_fontsize)
    ax_acc.set_ylabel("Accuracy", fontfamily=args.curve_font_family, fontsize=args.curve_axis_label_fontsize)
    ax_acc.set_ylim(0.0, 1.0)

    _style_axis_ticks(ax_loss, args.curve_font_family, args.curve_tick_fontsize)
    _style_axis_ticks(ax_acc, args.curve_font_family, args.curve_tick_fontsize)

    ax_loss.grid(alpha=0.25)
    handles = [line_train_loss, line_test_loss, line_train_acc, line_test_acc]
    labels = [h.get_label() for h in handles]
    legend_kwargs = dict(
        loc=args.curve_legend_loc,
        fontsize=args.curve_legend_fontsize,
        frameon=True,
        framealpha=framealpha,
    )
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    legend = ax_loss.legend(handles, labels, **legend_kwargs)
    if legend is not None:
        for t in legend.get_texts():
            t.set_fontfamily(args.curve_font_family)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset_name: str) -> int:
    return 10 if dataset_name == "cifar10" else 100


def linear_warmup_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target)
    scale = min(1.0, float(epoch) / float(warmup_epochs))
    return float(target) * scale


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], float(lam)


def build_cosine_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = max(int(warmup_epochs), 0)

    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        if epochs <= warmup_epochs:
            return 1.0
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin_mem = str(args.device).startswith("cuda")
    mean = CIFAR_MEAN
    std = CIFAR_STD

    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.auto_augment and hasattr(transforms, "AutoAugment"):
        train_ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if args.random_erasing > 0:
        train_ops.append(transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    train_tf = transforms.Compose(train_ops)
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    num_classes = get_num_classes(args.dataset)
    if args.fake_data:
        train_set = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_tf,
        )
        train_eval_set = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=eval_tf,
        )
        test_set = datasets.FakeData(
            size=args.fake_test_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=eval_tf,
        )
    else:
        dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train_set = dataset_cls(root=args.data_root, train=True, transform=train_tf, download=args.download)
        train_eval_set = dataset_cls(root=args.data_root, train=True, transform=eval_tf, download=args.download)
        test_set = dataset_cls(root=args.data_root, train=False, transform=eval_tf, download=args.download)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, train_eval_loader, test_loader


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()


def macro_f1_from_confusion(conf_mat: torch.Tensor, eps: float = 1e-12) -> float:
    conf = conf_mat.to(dtype=torch.float64)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    support = conf.sum(dim=1)
    valid = support > 0
    if torch.any(valid):
        return float(f1[valid].mean().item())
    return 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    xi_current: float,
    mu_current: float,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    task_sum = 0.0
    const_sum = 0.0
    fdgc_sum = 0.0
    acc_sum = 0.0
    alpha_sum = 0.0
    n_batches = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        mixed_images, labels_a, labels_b, lam = mixup_batch(images, labels, args.mixup_alpha)
        logits, aux = model(mixed_images, return_aux=True)
        if args.mixup_alpha > 0:
            mixed_task_loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
        else:
            mixed_task_loss = criterion(logits, labels)

        total_loss, losses = fdgc_total_loss(
            logits=logits,
            labels=labels,
            aux=aux,
            xi=xi_current,
            mu=mu_current,
            task_loss=mixed_task_loss,
        )# total_loss用来 backward()回传训练 losses是分项损失，主要用于打印和记录
        total_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        loss_sum += total_loss.item()
        task_sum += losses["task_loss"].item()
        const_sum += losses["const_loss"].item()
        fdgc_sum += losses["fdgc_loss"].item()
        if args.mixup_alpha > 0:
            mixup_acc = lam * batch_accuracy(logits, labels_a) + (1.0 - lam) * batch_accuracy(logits, labels_b)
        else:
            mixup_acc = batch_accuracy(logits, labels)
        acc_sum += mixup_acc
        alpha_sum += aux["alphas"].mean().item()
        n_batches += 1

        if step % args.print_freq == 0:
            print(
                f"[Train][Epoch {epoch}] Step {step}/{len(loader)} "
                f"loss={total_loss.item():.4f} task={losses['task_loss'].item():.4f} "
                f"const={losses['const_loss'].item():.4f} fdgc={losses['fdgc_loss'].item():.4f} "
                f"mixup_acc={mixup_acc:.4f} xi={xi_current:.5f} mu={mu_current:.5f}"
            )

    return {
        "loss": loss_sum / max(n_batches, 1),
        "task_loss": task_sum / max(n_batches, 1),
        "const_loss": const_sum / max(n_batches, 1),
        "fdgc_loss": fdgc_sum / max(n_batches, 1),
        "acc": acc_sum / max(n_batches, 1),
        "alpha": alpha_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args,
    xi_current: float,
    mu_current: float,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    task_sum = 0.0
    const_sum = 0.0
    fdgc_sum = 0.0
    acc_sum = 0.0
    alpha_sum = 0.0
    n_batches = 0
    conf_mat = None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, aux = model(images, return_aux=True)
        total_loss, losses = fdgc_total_loss(
            logits=logits,
            labels=labels,
            aux=aux,
            task_loss_fn=criterion,
            xi=xi_current,
            mu=mu_current,
        )

        loss_sum += total_loss.item()
        task_sum += losses["task_loss"].item()
        const_sum += losses["const_loss"].item()
        fdgc_sum += losses["fdgc_loss"].item()
        acc_sum += batch_accuracy(logits, labels)
        alpha_sum += aux["alphas"].mean().item()
        pred = torch.argmax(logits, dim=1)
        num_classes = logits.size(1)
        if conf_mat is None:
            conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        pair_index = labels.detach().cpu().to(torch.int64) * num_classes + pred.detach().cpu().to(torch.int64)
        conf_mat += torch.bincount(pair_index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        n_batches += 1

    macro_f1 = macro_f1_from_confusion(conf_mat) if conf_mat is not None else 0.0
    return {
        "loss": loss_sum / max(n_batches, 1),
        "task_loss": task_sum / max(n_batches, 1),
        "const_loss": const_sum / max(n_batches, 1),
        "fdgc_loss": fdgc_sum / max(n_batches, 1),
        "acc": acc_sum / max(n_batches, 1),
        "f1": macro_f1,
        "alpha": alpha_sum / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    acc_sum = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, return_aux=False)
        acc_sum += batch_accuracy(logits, labels)
        n_batches += 1
    return acc_sum / max(n_batches, 1)


def get_class_names(loader: DataLoader, num_classes: int) -> List[str]:
    dataset = getattr(loader, "dataset", None)
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (list, tuple)) and len(classes) == num_classes:
        return [str(c) for c in classes]
    return [f"C{i}" for i in range(num_classes)]


def _balanced_subsample(features: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, labels
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    per_class = max(1, max_points // max(1, len(classes)))
    keep_idx = []
    for c in classes:
        idx = np.where(labels == c)[0]
        take = min(len(idx), per_class)
        if take > 0:
            keep_idx.append(rng.choice(idx, size=take, replace=False))
    if keep_idx:
        keep_idx = np.concatenate(keep_idx)
    else:
        keep_idx = rng.choice(features.shape[0], size=max_points, replace=False)
    if keep_idx.size < max_points:
        remaining = np.setdiff1d(np.arange(features.shape[0]), keep_idx)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(max_points - keep_idx.size, remaining.size), replace=False)
            keep_idx = np.concatenate([keep_idx, extra])
    return features[keep_idx], labels[keep_idx]


def _reduce_2d(features: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    if features.shape[1] <= 2:
        if features.shape[1] == 1:
            return np.concatenate([features, np.zeros((features.shape[0], 1), dtype=features.dtype)], axis=1)
        return features

    if TSNE is not None and features.shape[0] >= 3:
        p = min(max(2, int(perplexity)), max(2, features.shape[0] - 1))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=p,
            random_state=seed,
        ).fit_transform(features)

    x = features - features.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    z = u[:, :2] * s[:2]
    if z.shape[1] < 2:
        z = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=z.dtype)], axis=1)
    return z


def save_embedding_2d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    figure_path: str,
    args,
):
    model.eval()
    feature_batches = []
    label_batches = []
    captured = {"feat": None}

    def hook_fn(_, hook_input, __):
        captured["feat"] = hook_input[0].detach()

    hook_handle = model.classifier[0].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                _ = model(images, return_aux=False)
                feat = captured.get("feat", None)
                if feat is None:
                    continue
                feature_batches.append(feat.cpu())
                label_batches.append(labels.cpu())
    finally:
        hook_handle.remove()

    if not feature_batches:
        return

    features = torch.cat(feature_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
    features, labels = _balanced_subsample(features, labels, args.embed_max_points, seed=args.seed)
    emb2d = _reduce_2d(features, perplexity=args.embed_perplexity, seed=args.seed)

    legend_bbox = parse_legend_bbox(args.embed_legend_bbox)
    framealpha = _clamp_01(args.embed_legend_framealpha)
    point_alpha = _clamp_01(args.embed_point_alpha)
    point_size = max(float(args.embed_point_size), 1.0)

    uniq = np.unique(labels)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    cmap = plt.get_cmap("tab20", len(uniq)) if len(uniq) <= 20 else plt.get_cmap("nipy_spectral", len(uniq))
    for i, c in enumerate(uniq):
        idx = labels == c
        name = class_names[int(c)] if 0 <= int(c) < len(class_names) else f"C{int(c)}"
        ax.scatter(
            emb2d[idx, 0],
            emb2d[idx, 1],
            s=point_size,
            alpha=point_alpha,
            color=cmap(i),
            label=name,
            edgecolors="none",
        )

    ax.set_title("2D Embedding (Test Features)", fontfamily=args.embed_font_family, fontsize=args.embed_title_fontsize)
    ax.set_xlabel("Dim-1", fontfamily=args.embed_font_family, fontsize=args.embed_axis_label_fontsize)
    ax.set_ylabel("Dim-2", fontfamily=args.embed_font_family, fontsize=args.embed_axis_label_fontsize)
    _style_axis_ticks(ax, args.embed_font_family, args.embed_tick_fontsize)
    ax.grid(alpha=0.2)

    if len(uniq) <= max(1, int(args.embed_max_legend_classes)):
        legend_kwargs = dict(
            loc=args.embed_legend_loc,
            fontsize=args.embed_legend_fontsize,
            frameon=True,
            framealpha=framealpha,
            title="Class",
        )
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        legend = ax.legend(**legend_kwargs)
        if legend is not None:
            for t in legend.get_texts():
                t.set_fontfamily(args.embed_font_family)
            if legend.get_title() is not None:
                legend.get_title().set_fontfamily(args.embed_font_family)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def _denormalize_cifar(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def save_gradcam_style_attention_map(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    figure_path: str,
    args,
):
    num_images = max(int(args.gradcam_num_images), 0)
    # For side-by-side comparison: keep 4 samples max (row-1 original, row-2 heatmap overlay).
    num_images = min(num_images, 4)
    if num_images <= 0:
        return

    if getattr(model, "backbone_name", "") == "resnet18":
        target_module = model.backbone.layer4
    else:
        target_module = model.stem

    cached = {"acts": None, "grads": None}

    def forward_hook(_, __, output):
        cached["acts"] = output

    def backward_hook(_, __, grad_output):
        cached["grads"] = grad_output[0]

    h_fwd = target_module.register_forward_hook(forward_hook)
    h_bwd = target_module.register_full_backward_hook(backward_hook)

    overlays = []
    gt_labels = []
    pred_labels = []
    alpha = _clamp_01(args.gradcam_alpha)
    model.eval()

    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            logits = model(images, return_aux=False)
            pred = torch.argmax(logits, dim=1)
            chosen = logits.gather(1, pred.unsqueeze(1)).sum()
            chosen.backward()

            acts = cached.get("acts", None)
            grads = cached.get("grads", None)
            if acts is None or grads is None:
                continue

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1))
            cam = cam - cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
            cam = cam / (cam.flatten(1).max(dim=1)[0].view(-1, 1, 1) + 1e-6)
            cam = F.interpolate(cam.unsqueeze(1), size=(images.size(2), images.size(3)), mode="bilinear", align_corners=False)
            cam = cam.squeeze(1).detach().cpu()
            image_vis = _denormalize_cifar(images.detach()).cpu()

            batch_size = images.size(0)
            for i in range(batch_size):
                overlays.append((image_vis[i], cam[i]))
                gt_labels.append(int(labels[i].item()))
                pred_labels.append(int(pred[i].item()))
                if len(overlays) >= num_images:
                    break
            if len(overlays) >= num_images:
                break
    finally:
        h_fwd.remove()
        h_bwd.remove()

    if not overlays:
        return

    n = len(overlays)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.0))
    axes = np.atleast_2d(axes)

    for i in range(n):
        image_t, cam_t = overlays[i]
        image_np = image_t.permute(1, 2, 0).numpy()
        cam_np = cam_t.numpy()
        gt = gt_labels[i]
        pred = pred_labels[i]
        gt_name = class_names[gt] if 0 <= gt < len(class_names) else f"C{gt}"
        pred_name = class_names[pred] if 0 <= pred < len(class_names) else f"C{pred}"

        ax_top = axes[0, i]
        ax_top.imshow(image_np)
        ax_top.set_title(f"Original\nGT:{gt_name} | Pred:{pred_name}", fontsize=args.gradcam_title_fontsize)
        ax_top.axis("off")

        ax_bottom = axes[1, i]
        ax_bottom.imshow(image_np)
        ax_bottom.imshow(cam_np, cmap="jet", alpha=alpha)
        ax_bottom.set_title("Grad-CAM Overlay", fontsize=args.gradcam_title_fontsize)
        ax_bottom.axis("off")

    fig.suptitle("Grad-CAM Comparison (Top: Original | Bottom: Heatmap)", fontsize=args.gradcam_suptitle_fontsize)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.save_dir = resolve_save_dir(args.save_dir)
    run_paths = build_run_paths(args)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_task": [],
        "train_const": [],
        "train_fdgc": [],
        "train_mixup_acc": [],
        "train_clean_acc": [],
        "test_loss": [],
        "test_task": [],
        "test_const": [],
        "test_fdgc": [],
        "test_acc": [],
        "test_f1": [],
        "alpha": [],
        "xi": [],
        "mu": [],
        "lr": [],
    }
    model = None
    test_loader = None
    class_names = None

    log_fp = open(run_paths["log_path"], "a", encoding="utf-8", buffering=1)
    stdout_backup, stderr_backup = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_fp)
    sys.stderr = Tee(sys.stderr, log_fp)

    try:
        print(f"Command: {sys.executable} {' '.join(sys.argv)}")
        print(f"Resolved save_dir: {args.save_dir}")
        print(f"Run directory: {run_paths['run_dir']}")
        print(f"Log file: {run_paths['log_path']}")

        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                if args.strict_cuda:
                    raise RuntimeError("CUDA device requested but not available. Use --allow_cpu_fallback to run on CPU.")
                print("CUDA is not available, switch to CPU.")
                args.device = "cpu"
            else:
                gpu_count = torch.cuda.device_count()
                if args.gpu_id is not None:
                    selected_gpu = args.gpu_id
                else:
                    if ":" in args.device:
                        try:
                            selected_gpu = int(args.device.split(":")[1])
                        except ValueError as e:
                            raise ValueError(f"Invalid --device value '{args.device}'. Expected format cuda or cuda:<index>.") from e
                    else:
                        selected_gpu = 0

                if selected_gpu < 0 or selected_gpu >= gpu_count:
                    raise ValueError(f"Invalid --gpu_id={selected_gpu}. Available GPU indices: 0..{gpu_count-1}")

                torch.cuda.set_device(selected_gpu)
                args.device = f"cuda:{selected_gpu}"
                torch.backends.cudnn.benchmark = True
                print(f"CUDA available: using GPU {selected_gpu}/{gpu_count-1}: {torch.cuda.get_device_name(selected_gpu)}")
        device = torch.device(args.device)

        train_loader, train_eval_loader, test_loader = build_dataloaders(args)
        num_classes = get_num_classes(args.dataset)
        class_names = get_class_names(test_loader, num_classes)
        model = FDGCCIFARNet(
            in_channels=3,
            num_classes=num_classes,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            dictionary_size=args.dictionary_size,
            num_stages=args.num_stages,
            gamma_a=args.gamma_a,
            gamma_s=args.gamma_s,
            rho=args.rho,
            topk_atom=args.topk_atom,
            topk_token=args.topk_token,
            sigma_atom=args.sigma_atom,
            sigma_token=args.sigma_token,
            transform_hidden=args.transform_hidden,
            backbone=args.backbone,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )

        best_acc = 0.0
        print(f"Start training on {args.dataset}, device={device}, epochs={args.epochs}")
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            xi_current = linear_warmup_weight(args.xi, epoch, args.fdgc_warmup_epochs)
            mu_current = linear_warmup_weight(args.mu, epoch, args.fdgc_warmup_epochs)
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                args=args,
                xi_current=xi_current,
                mu_current=mu_current,
            )
            test_stats = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                args=args,
                xi_current=args.xi,
                mu_current=args.mu,
            )
            train_clean_acc = evaluate_accuracy(model, train_eval_loader, device)
            scheduler.step()

            history["epoch"].append(epoch)
            history["train_loss"].append(train_stats["loss"])
            history["train_task"].append(train_stats["task_loss"])
            history["train_const"].append(train_stats["const_loss"])
            history["train_fdgc"].append(train_stats["fdgc_loss"])
            history["train_mixup_acc"].append(train_stats["acc"])
            history["train_clean_acc"].append(train_clean_acc)
            history["test_loss"].append(test_stats["loss"])
            history["test_task"].append(test_stats["task_loss"])
            history["test_const"].append(test_stats["const_loss"])
            history["test_fdgc"].append(test_stats["fdgc_loss"])
            history["test_acc"].append(test_stats["acc"])
            history["test_f1"].append(test_stats["f1"])
            history["alpha"].append(test_stats["alpha"])
            history["xi"].append(xi_current)
            history["mu"].append(mu_current)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_stats['loss']:.4f} train_mixup_acc={train_stats['acc']:.4f} "
                f"train_clean_acc={train_clean_acc:.4f} "
                f"test_loss={test_stats['loss']:.4f} test_acc={test_stats['acc']:.4f} test_f1={test_stats['f1']:.4f} "
                f"alpha~{test_stats['alpha']:.4f} lr={optimizer.param_groups[0]['lr']:.6f} "
                f"xi={xi_current:.5f} mu={mu_current:.5f}"
            )

            if test_stats["acc"] > best_acc:
                best_acc = test_stats["acc"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "args": vars(args),
                        "run_name": run_paths["run_name"],
                    },
                    run_paths["best_path"],
                )
                print(f"Saved best checkpoint to {run_paths['best_path']} (acc={best_acc:.4f})")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                    "run_name": run_paths["run_name"],
                },
                run_paths["last_path"],
            )

        elapsed = time.time() - t0
        print(f"Training done. Best test acc={best_acc:.4f}, elapsed={elapsed / 60.0:.2f} min")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current logs and curves...")
    finally:
        try:
            save_history_csv(history, run_paths["metric_csv_path"])
        except Exception as e:
            print(f"[Warn] Failed to save metrics CSV: {e}")
        try:
            save_convergence_curve(history, run_paths["curve_path"], args)
        except Exception as e:
            print(f"[Warn] Failed to save convergence curve: {e}")
        if model is not None and test_loader is not None and class_names is not None:
            if args.save_embedding_vis:
                try:
                    save_embedding_2d(
                        model=model,
                        loader=test_loader,
                        device=torch.device(args.device),
                        class_names=class_names,
                        figure_path=run_paths["embedding_path"],
                        args=args,
                    )
                    print(f"Saved 2D embedding figure to: {run_paths['embedding_path']}")
                except Exception as e:
                    print(f"[Warn] Failed to save 2D embedding figure: {e}")
            if args.save_gradcam_vis:
                try:
                    save_gradcam_style_attention_map(
                        model=model,
                        loader=test_loader,
                        device=torch.device(args.device),
                        class_names=class_names,
                        figure_path=run_paths["gradcam_path"],
                        args=args,
                    )
                    print(f"Saved Grad-CAM figure to: {run_paths['gradcam_path']}")
                except Exception as e:
                    print(f"[Warn] Failed to save Grad-CAM figure: {e}")
        print(f"Saved metrics CSV to: {run_paths['metric_csv_path']}")
        print(f"Saved convergence curve to: {run_paths['curve_path']}")
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        log_fp.close()


if __name__ == "__main__":
    main()
