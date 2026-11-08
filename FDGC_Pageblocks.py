

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-6


def default_pageblocks_data_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    return os.path.join(project_dir, "data", "page-blocks.data")


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


def build_log_path(dataset_name: str, log_dir: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = dataset_name.replace(" ", "_")
    if not log_dir:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{safe_name}-{timestamp}.log")


def build_plot_prefix(run_name: str, plot_dir: str = "") -> str:
    if not plot_dir:
        plot_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = run_name.replace(" ", "_")
    return os.path.join(plot_dir, safe_name)


def parse_legend_bbox(legend_bbox: str) -> Optional[Tuple[float, float]]:
    text = (legend_bbox or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid --legend_bbox='{legend_bbox}', expected format: x,y (e.g. 0.02,0.98)")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid --legend_bbox='{legend_bbox}', expected numeric x,y values") from exc


def save_training_curve(total_loss: np.ndarray, train_acc: np.ndarray, test_acc: np.ndarray, dataset_name: str, save_path: str):
    epochs = np.arange(1, len(total_loss) + 1)
    fig = plt.figure(figsize=(8.2, 5.2), dpi=150)
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, total_loss, color="#d62728", linewidth=2.0, label="total_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, color="#1f77b4", linewidth=2.0, label="train_acc")
    ax2.plot(epochs, test_acc, color="#2ca02c", linewidth=2.0, label="test_acc")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)
    ax1.set_title(f"{dataset_name} - Training Curve (total_loss / train_acc / test_acc)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_embedding_scatter(
    code: np.ndarray,
    label: np.ndarray,
    fold_id: np.ndarray,
    dataset_name: str,
    save_path: str,
    legend_loc: str = "best",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_fontsize: float = 8.0,
    legend_title_fontsize: float = 9.0,
    legend_framealpha: float = 0.9,
    legend_text_alpha: float = 1.0,
    legend_ncol: int = 1,
    scatter_alpha: float = 0.85,
):
    code = np.asarray(code, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)
    fold_id = np.asarray(fold_id, dtype=np.int64)
    if (
        code.ndim != 2
        or label.ndim != 1
        or fold_id.ndim != 1
        or code.shape[0] != label.shape[0]
        or code.shape[0] != fold_id.shape[0]
        or code.shape[0] < 2
    ):
        return

    rng = np.random.RandomState(42)
    max_points = 3000
    if code.shape[0] > max_points:
        pair = np.stack([label, fold_id], axis=1)
        unique_pair = np.unique(pair, axis=0)
        per_pair = max(1, max_points // max(len(unique_pair), 1))
        keep_idx = []
        for c, f in unique_pair:
            idx = np.where((label == c) & (fold_id == f))[0]
            take = min(len(idx), per_pair)
            if take > 0:
                keep_idx.append(rng.choice(idx, size=take, replace=False))
        if keep_idx:
            keep_idx = np.concatenate(keep_idx)
        else:
            keep_idx = rng.choice(code.shape[0], size=max_points, replace=False)
        if keep_idx.shape[0] < max_points:
            remaining = np.setdiff1d(np.arange(code.shape[0]), keep_idx)
            if remaining.size > 0:
                extra = rng.choice(remaining, size=min(max_points - keep_idx.shape[0], remaining.size), replace=False)
                keep_idx = np.concatenate([keep_idx, extra])
        code = code[keep_idx]
        label = label[keep_idx]
        fold_id = fold_id[keep_idx]

    if code.shape[1] >= 2:
        perplexity = min(30, max(5, code.shape[0] // 20))
        perplexity = min(perplexity, max(1, code.shape[0] - 1))
        if code.shape[0] >= 3 and perplexity >= 2:
            z2d = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42).fit_transform(code)
        else:
            z2d = code[:, :2]
    else:
        z2d = np.concatenate([code, np.zeros((code.shape[0], 1), dtype=np.float32)], axis=1)

    classes = np.unique(label)
    folds = np.unique(fold_id)
    scatter_alpha = float(np.clip(scatter_alpha, 0.0, 1.0))
    legend_framealpha = float(np.clip(legend_framealpha, 0.0, 1.0))
    legend_text_alpha = float(np.clip(legend_text_alpha, 0.0, 1.0))
    legend_ncol = max(1, int(legend_ncol))
    fig = plt.figure(figsize=(7.5, 5.8), dpi=150)
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab20", len(classes))
    marker_pool = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    marker_map = {f: marker_pool[i % len(marker_pool)] for i, f in enumerate(folds)}
    for i, c in enumerate(classes):
        for f in folds:
            idx = (label == c) & (fold_id == f)
            if not np.any(idx):
                continue
            ax.scatter(
                z2d[idx, 0],
                z2d[idx, 1],
                s=22,
                alpha=scatter_alpha,
                color=cmap(i),
                marker=marker_map[f],
                label=f"class={int(c)}, fold={int(f)}",
                linewidths=0.15,
                edgecolors="black",
            )
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.set_title(f"{dataset_name} - 2D Embedding (color=class, marker=fold)")
    if len(classes) * len(folds) <= 30:
        legend_kwargs = dict(
            title="Class / Fold",
            loc=legend_loc,
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            framealpha=legend_framealpha,
            ncol=legend_ncol,
        )
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        leg = ax.legend(**legend_kwargs)
        if leg is not None:
            for text in leg.get_texts():
                text.set_alpha(legend_text_alpha)
            if leg.get_title() is not None:
                leg.get_title().set_alpha(legend_text_alpha)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def build_knn_affinity(features: torch.Tensor, topk: int, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), EPS)
    dist2 = torch.cdist(features, features, p=2).pow(2)
    affinity = torch.exp(-dist2 / (sigma * sigma))
    affinity.fill_diagonal_(0.0)

    n = affinity.size(0)
    k = min(max(int(topk), 0), max(n - 1, 0))
    if k > 0:
        _, idx = torch.topk(affinity, k=k, dim=-1)
        mask = torch.zeros_like(affinity)
        mask.scatter_(1, idx, 1.0)
        affinity = affinity * mask

    return 0.5 * (affinity + affinity.t())


def normalized_laplacian(affinity: torch.Tensor) -> torch.Tensor:
    degree = affinity.sum(dim=-1).clamp_min(EPS)
    d_inv_sqrt = degree.rsqrt()
    normalized_adj = d_inv_sqrt.unsqueeze(1) * affinity * d_inv_sqrt.unsqueeze(0)
    eye = torch.eye(affinity.size(0), device=affinity.device, dtype=affinity.dtype)
    return eye - normalized_adj


def fractional_power_laplacian(laplacian: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    laplacian = 0.5 * (laplacian + laplacian.t())
    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    eigvals = eigvals.clamp_min(0.0) + EPS
    eigvals_alpha = eigvals.pow(alpha)
    return (eigvecs * eigvals_alpha.unsqueeze(0)) @ eigvecs.t()


def _parse_target_index(target_col: str, n_cols: int) -> int:
    try:
        idx = int(target_col)
    except ValueError as exc:
        raise ValueError(f"target_col must be integer-like for page-blocks.data, got '{target_col}'") from exc
    if idx < 0:
        idx = n_cols + idx
    if idx < 0 or idx >= n_cols:
        raise ValueError(f"target_col index out of range: {target_col}, n_cols={n_cols}")
    return idx


def load_pageblocks_data(data_path: str, target_col: str = "-1") -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Pageblocks dataset is whitespace-separated without header.
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        engine="python",
        na_values=["?", "NA", "nan", "None", ""],
    )
    target_idx = _parse_target_index(target_col, df.shape[1])

    y = pd.to_numeric(df.iloc[:, target_idx], errors="coerce")
    x_df = df.drop(columns=[target_idx]).copy()
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    valid_label = ~y.isna()
    x_df = x_df.loc[valid_label].copy()
    y = y.loc[valid_label].copy()

    x_df = x_df.fillna(x_df.median(numeric_only=True))

    y_np = y.to_numpy()
    _, y_encoded = np.unique(y_np, return_inverse=True)
    y_encoded = y_encoded.astype(np.int64)
    x_np = x_df.to_numpy(dtype=np.float32)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Detected fewer than 2 classes after preprocessing.")
    return x_np, y_encoded


class FDGCTabularClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dictionary_size: int = 32,
        gamma_a: float = 0.05,
        gamma_s: float = 0.05,
        rho: float = 1.0,
        topk_atom: int = 8,
        topk_sample: int = 8,
        sigma_atom: float = 1.0,
        sigma_sample: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gamma_a = float(gamma_a)
        self.gamma_s = float(gamma_s)
        self.rho = float(rho)
        self.topk_atom = int(topk_atom)
        self.topk_sample = int(topk_sample)
        self.sigma_atom = float(sigma_atom)
        self.sigma_sample = float(sigma_sample)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.code_proj = nn.Linear(hidden_dim, dictionary_size)
        self.classifier = nn.Linear(dictionary_size, num_classes)

        self.transform = nn.Linear(dictionary_size, dictionary_size, bias=False)
        self.inverse_transform = nn.Linear(dictionary_size, dictionary_size, bias=False)
        self.tau_raw = nn.Parameter(torch.tensor(0.02))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # alpha in (1,2)

        self.dictionary = nn.Parameter(torch.empty(hidden_dim, dictionary_size))
        self.register_buffer("dictionary_initialized", torch.tensor(False), persistent=False)

    def _initialize_dictionary_from_features(self, feat: torch.Tensor):
        if bool(self.dictionary_initialized):
            return
        with torch.no_grad():
            atoms = F.normalize(feat.detach(), dim=1)
            num_atoms = self.dictionary.size(1)
            if atoms.size(0) >= num_atoms:
                idx = torch.randperm(atoms.size(0), device=atoms.device)[:num_atoms]
                init_atoms = atoms[idx]
            else:
                repeats = (num_atoms + atoms.size(0) - 1) // atoms.size(0)
                init_atoms = atoms.repeat(repeats, 1)[:num_atoms]
            self.dictionary.copy_(init_atoms.t().contiguous())
            self.dictionary_initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        code = F.relu(self.code_proj(feat))
        logits = self.classifier(code)
        return logits, code, feat

    def fdgc_losses(self, code: torch.Tensor, feat: torch.Tensor):
        self._initialize_dictionary_from_features(feat)
        alpha = 1.0 + torch.sigmoid(self.alpha_raw)
        tau = F.softplus(self.tau_raw) + EPS

        atom_feat = F.normalize(self.dictionary.detach().t(), dim=1)
        atom_affinity = build_knn_affinity(atom_feat, self.topk_atom, self.sigma_atom)
        lap_atom_alpha = fractional_power_laplacian(normalized_laplacian(atom_affinity), alpha)

        sample_feat = F.normalize(feat.detach(), dim=1)
        sample_affinity = build_knn_affinity(sample_feat, self.topk_sample, self.sigma_sample)
        lap_sample_alpha = fractional_power_laplacian(normalized_laplacian(sample_affinity), alpha)

        atom_energy = torch.einsum("bm,mn,bn->b", code, lap_atom_alpha, code).mean()
        sample_energy = torch.trace(code.t() @ lap_sample_alpha @ code) / max(code.size(0), 1)
        fdgc_loss = self.gamma_a * atom_energy + self.gamma_s * self.rho * sample_energy

        z = self.transform(code)
        z_soft = soft_threshold(z, tau)
        code_rec = F.relu(self.inverse_transform(z_soft))
        const_loss = F.mse_loss(code_rec, code)
        return const_loss, fdgc_loss, float(alpha.detach().cpu())


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    best_test_acc: float
    best_test_f1: float
    best_epoch: int
    epoch_total_loss: List[float]
    epoch_train_acc: List[float]
    epoch_test_acc: List[float]
    test_code: np.ndarray
    test_label: np.ndarray


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            logits, _, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


def collect_codes_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    codes, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))
            _, code, _ = model(x)
            codes.append(code.detach().cpu())
            labels.append(y.detach().cpu())
    if not codes:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return torch.cat(codes, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def train_one_fold(
    fold_id: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args,
    device: torch.device,
) -> FoldResult:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    model = FDGCTabularClassifier(
        input_dim=x_train.shape[1],
        num_classes=len(np.unique(y_train)),
        hidden_dim=args.hidden_dim,
        dictionary_size=args.dictionary_size,
        gamma_a=args.gamma_a,
        gamma_s=args.gamma_s,
        rho=args.rho,
        topk_atom=args.topk_atom,
        topk_sample=args.topk_sample,
        sigma_atom=args.sigma_atom,
        sigma_sample=args.sigma_sample,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_acc, best_f1, best_epoch = 0.0, 0.0, 0
    epoch_total_hist: List[float] = []
    epoch_train_acc_hist: List[float] = []
    epoch_test_acc_hist: List[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total, epoch_task, epoch_const, epoch_fdgc = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        train_correct, train_count = 0, 0

        scale = min(1.0, epoch / max(args.fdgc_warmup_epochs, 1))
        xi_cur = args.xi * scale
        mu_cur = args.mu * scale

        for x, y in train_loader:
            x = x.to(device, non_blocking=pin_mem)
            y = y.to(device, non_blocking=pin_mem)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits, code, feat = model(x)
                task_loss = criterion(logits, y)

            const_loss, fdgc_loss, alpha = model.fdgc_losses(code.float(), feat.float())
            total_loss = task_loss + xi_cur * const_loss + mu_cur * fdgc_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_total += total_loss.item()
            epoch_task += task_loss.item()
            epoch_const += const_loss.item()
            epoch_fdgc += fdgc_loss.item()
            n_batches += 1
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == y).sum().item()
            train_count += y.size(0)

        test_acc, test_f1 = evaluate_model(model, test_loader, device)
        train_acc = train_correct / max(train_count, 1)
        epoch_total_hist.append(epoch_total / max(n_batches, 1))
        epoch_train_acc_hist.append(train_acc)
        epoch_test_acc_hist.append(test_acc)
        if test_acc > best_acc:
            best_acc, best_f1, best_epoch = test_acc, test_f1, epoch

        if epoch % args.print_freq == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[Fold {fold_id}] Epoch {epoch:03d} "
                f"loss={epoch_total/max(n_batches,1):.4f} "
                f"task={epoch_task/max(n_batches,1):.4f} "
                f"const={epoch_const/max(n_batches,1):.4f} "
                f"fdgc={epoch_fdgc/max(n_batches,1):.4f} "
                f"train_acc={train_acc:.4f} alpha~{alpha:.4f} xi={xi_cur:.5f} mu={mu_cur:.5f} "
                f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
            )

    test_code, test_label = collect_codes_labels(model, test_loader, device)
    return FoldResult(
        fold=fold_id,
        train_size=len(train_ds),
        test_size=len(test_ds),
        best_test_acc=best_acc,
        best_test_f1=best_f1,
        best_epoch=best_epoch,
        epoch_total_loss=epoch_total_hist,
        epoch_train_acc=epoch_train_acc_hist,
        epoch_test_acc=epoch_test_acc_hist,
        test_code=test_code,
        test_label=test_label,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="FDGC-style Pageblocks classification with 5-fold CV")
    parser.add_argument("--data_path", type=str, default=default_pageblocks_data_path(), help="Path to page-blocks.data")
    parser.add_argument(
        "--target_col",
        type=str,
        default="-1",
        help="Label column index for page-blocks.data (default -1 means last column)",
    )
    parser.add_argument("--log_dir", type=str, default="", help="Directory to save log file")
    parser.add_argument("--plot_dir", type=str, default="", help="Directory to save visualization figures")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=710)
    # 42:0.9761, 3407:0.9764, 114514:0.9761, 256:0.9757, 710:0.9762
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable CUDA AMP mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--print_freq", type=int, default=20)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dictionary_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)

    parser.add_argument("--gamma_a", type=float, default=0.05)
    parser.add_argument("--gamma_s", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.005)
    parser.add_argument("--fdgc_warmup_epochs", type=int, default=40)
    parser.add_argument("--topk_atom", type=int, default=8)
    parser.add_argument("--topk_sample", type=int, default=8)
    parser.add_argument("--sigma_atom", type=float, default=1.0)
    parser.add_argument("--sigma_sample", type=float, default=1.0)
    parser.add_argument("--legend_loc", type=str, default="best", help="Legend location, e.g. best/upper right/lower left")
    parser.add_argument("--legend_bbox", type=str, default="", help="Optional legend bbox anchor as 'x,y', e.g. 0.02,0.98")
    parser.add_argument("--legend_fontsize", type=float, default=8.0, help="Legend text font size")
    parser.add_argument("--legend_title_fontsize", type=float, default=9.0, help="Legend title font size")
    parser.add_argument("--legend_framealpha", type=float, default=0.90, help="Legend frame alpha in [0,1]")
    parser.add_argument("--legend_text_alpha", type=float, default=1.00, help="Legend text alpha in [0,1]")
    parser.add_argument("--legend_ncol", type=int, default=1, help="Legend column count")
    parser.add_argument("--scatter_alpha", type=float, default=0.85, help="Scatter point alpha in [0,1]")
    return parser.parse_args()


def core_main(args):
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Please run on a machine with NVIDIA GPU.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    dataset_name = "Pageblocks"
    x, y = load_pageblocks_data(args.data_path, args.target_col)
    num_classes = len(np.unique(y))
    print(f"Dataset: {dataset_name}, samples={len(x)}, features={x.shape[1]}, classes={num_classes}")
    print(f"Data path: {args.data_path}")
    print("Cross-validation: StratifiedKFold(n_splits=5, shuffle=True)")
    print(f"Each fold uses train:test = 4:1 ({int(len(x)*0.8)}:{int(len(x)*0.2)} approx.)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_results: List[FoldResult] = []
    fold_total_hist: List[List[float]] = []
    fold_train_acc_hist: List[List[float]] = []
    fold_test_acc_hist: List[List[float]] = []
    all_codes: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_folds: List[np.ndarray] = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        x_train_raw, y_train = x[train_idx], y[train_idx]
        x_test_raw, y_test = x[test_idx], y[test_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
        x_test = scaler.transform(x_test_raw).astype(np.float32)

        print(
            f"\n===== Fold {fold_id}/5 ===== "
            f"train={len(train_idx)} test={len(test_idx)} ratio={len(train_idx)/len(test_idx):.1f}:1"
        )
        fold_result = train_one_fold(
            fold_id=fold_id,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            args=args,
            device=device,
        )
        fold_results.append(fold_result)
        fold_total_hist.append(fold_result.epoch_total_loss)
        fold_train_acc_hist.append(fold_result.epoch_train_acc)
        fold_test_acc_hist.append(fold_result.epoch_test_acc)
        all_codes.append(fold_result.test_code)
        all_labels.append(fold_result.test_label)
        all_folds.append(np.full(fold_result.test_label.shape[0], fold_id, dtype=np.int64))
        print(
            f"[Fold {fold_id}] best_acc={fold_result.best_test_acc:.4f} "
            f"best_f1={fold_result.best_test_f1:.4f} best_epoch={fold_result.best_epoch}"
        )

    accs = np.array([r.best_test_acc for r in fold_results], dtype=np.float64)
    f1s = np.array([r.best_test_f1 for r in fold_results], dtype=np.float64)
    print("\n===== 5-Fold Summary =====")
    for r in fold_results:
        print(
            f"Fold {r.fold}: train={r.train_size}, test={r.test_size}, "
            f"best_acc={r.best_test_acc:.4f}, best_f1={r.best_test_f1:.4f}, epoch={r.best_epoch}"
        )
    print(
        f"Mean Acc={accs.mean():.4f} +/- {accs.std(ddof=1):.4f}, "
        f"Mean F1={f1s.mean():.4f} +/- {f1s.std(ddof=1):.4f}"
    )

    mean_total = np.mean(np.array(fold_total_hist, dtype=np.float64), axis=0)
    mean_train_acc = np.mean(np.array(fold_train_acc_hist, dtype=np.float64), axis=0)
    mean_test_acc = np.mean(np.array(fold_test_acc_hist, dtype=np.float64), axis=0)
    code_all = np.concatenate(all_codes, axis=0)
    label_all = np.concatenate(all_labels, axis=0)
    fold_all = np.concatenate(all_folds, axis=0)
    legend_bbox = parse_legend_bbox(args.legend_bbox)

    run_name = getattr(args, "run_name", f"{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    plot_prefix = build_plot_prefix(run_name, args.plot_dir)
    curve_path = f"{plot_prefix}-curve.png"
    embedding_path = f"{plot_prefix}-embedding.png"
    save_training_curve(mean_total, mean_train_acc, mean_test_acc, dataset_name, curve_path)
    save_embedding_scatter(
        code_all,
        label_all,
        fold_all,
        dataset_name,
        embedding_path,
        legend_loc=args.legend_loc,
        legend_bbox=legend_bbox,
        legend_fontsize=args.legend_fontsize,
        legend_title_fontsize=args.legend_title_fontsize,
        legend_framealpha=args.legend_framealpha,
        legend_text_alpha=args.legend_text_alpha,
        legend_ncol=args.legend_ncol,
        scatter_alpha=args.scatter_alpha,
    )
    print(f"Saved training curve: {curve_path}")
    print(f"Saved embedding scatter: {embedding_path}")


def run_with_logging():
    args = parse_args()
    dataset_name = "Pageblocks"
    log_path = build_log_path(dataset_name, args.log_dir)

    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    stdout_backup, stderr_backup = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_fp)
    sys.stderr = Tee(sys.stderr, log_fp)
    try:
        print(f"Command: {sys.executable} {' '.join(sys.argv)}")
        print(f"Log file: {log_path}")
        args.run_name = os.path.splitext(os.path.basename(log_path))[0]
        core_main(args)
    finally:
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        log_fp.close()


if __name__ == "__main__":
    run_with_logging()
