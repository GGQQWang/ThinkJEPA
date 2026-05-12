import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "cache_train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "cache_train"))
if str(REPO_ROOT / "vjepa2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "vjepa2"))

from cache_train.thinker_train import (  # noqa: E402
    align_frame_gt_to_latent_frames,
    build_frame_gt_index,
    build_sliding_latent_windows,
    build_thinkjepa_guidance_inputs,
    compute_latent_anomaly_scores,
    load_frame_gt_array,
    _normalize_thinker_cache_stem,
)
from cache_train.visualize_latent_anomaly_sample import (  # noqa: E402
    load_npz_sample,
    make_predictor,
    predict_window,
)


def parse_args():
    p = argparse.ArgumentParser(
        "Build a t-SNE plot for latent anomaly error embeddings."
    )
    p.add_argument("--cache_dir", required=True, help="Test cache root with .npz files.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path.")
    p.add_argument("--frame_gt_dir", required=True, help="Per-frame 0/1 GT label dir.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--past_T", type=int, default=32)
    p.add_argument("--future_T", type=int, default=16)
    p.add_argument("--anomaly_stride", type=int, default=16)
    p.add_argument("--anomaly_threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_files", type=int, default=0, help="0 means all files.")
    p.add_argument("--max_windows", type=int, default=5000)
    p.add_argument(
        "--embedding",
        choices=["abs_error", "signed_error", "target", "concat"],
        default="abs_error",
        help="Window vector used for t-SNE.",
    )
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_gt_for_cache(cache_path, frame_gt_index, frame_gt_cache):
    stem = _normalize_thinker_cache_stem(cache_path)
    label_path = frame_gt_index.get(stem)
    if label_path is None:
        return None
    if label_path not in frame_gt_cache:
        frame_gt_cache[label_path] = load_frame_gt_array(label_path)
    return frame_gt_cache[label_path]


def window_embedding(pred, target, mode):
    diff = pred.float() - target.float()
    if mode == "abs_error":
        return diff.abs().mean(dim=(1, 2))
    if mode == "signed_error":
        return diff.mean(dim=(1, 2))
    if mode == "target":
        return target.float().mean(dim=(1, 2))
    if mode == "concat":
        abs_err = diff.abs().mean(dim=(1, 2))
        tgt = target.float().mean(dim=(1, 2))
        score = torch.linalg.norm(diff, dim=-1).mean(dim=(1, 2), keepdim=True)
        return torch.cat([tgt, abs_err, score], dim=-1)
    raise ValueError(mode)


def build_plot(out_png, points, labels, scores, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 7))
    normal = labels == 0
    abnormal = labels == 1
    ax.scatter(
        points[normal, 0],
        points[normal, 1],
        s=12,
        c="#1f77b4",
        alpha=0.55,
        label=f"normal ({normal.sum()})",
        linewidths=0,
    )
    ax.scatter(
        points[abnormal, 0],
        points[abnormal, 1],
        s=18,
        c="#d62728",
        alpha=0.70,
        label=f"anomaly ({abnormal.sum()})",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    text = (
        f"score mean: normal={scores[normal].mean():.3f} "
        f"anomaly={scores[abnormal].mean():.3f}"
        if normal.any() and abnormal.any()
        else ""
    )
    if text:
        ax.text(0.01, 0.01, text, transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    files = sorted(glob.glob(os.path.join(args.cache_dir, "**", "*.npz"), recursive=True))
    if args.max_files > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {args.cache_dir}")

    frame_gt_index = build_frame_gt_index(args.frame_gt_dir)
    frame_gt_cache = {}
    if not frame_gt_index:
        raise FileNotFoundError(f"No GT label files indexed under {args.frame_gt_dir}")

    predictor = None
    embeddings = []
    labels = []
    scores = []
    rows = []

    guidance_args = SimpleNamespace(
        thinkjepa_use_vlm_merge=True,
        thinkjepa_vlm_source="both",
        thinkjepa_vlm_layer_selector="last",
        thinkjepa_vlm_layer_index=-1,
        thinkjepa_drop_thinking_tokens=False,
        thinkjepa_zero_dropped_think_tokens=True,
        thinkjepa_think_start_ids="151667",
        thinkjepa_think_end_ids="151668",
        thinkjepa_think_drop_ids="",
        thinkjepa_think_drop_prefix_len=0,
        thinkjepa_think_drop_suffix_len=0,
        thinkjepa_think_token_pad_id=-1,
        thinkjepa_think_prefix_open=False,
        thinkjepa_verbose=False,
    )

    for file_idx, cache_path in enumerate(files):
        if args.max_windows > 0 and len(labels) >= args.max_windows:
            break
        gt_raw = load_gt_for_cache(cache_path, frame_gt_index, frame_gt_cache)
        if gt_raw is None:
            continue
        try:
            feats, extras, frame_indices, video_path = load_npz_sample(cache_path, device)
        except Exception as exc:
            print(f"[WARN] skip unreadable cache: {cache_path} ({type(exc).__name__}: {exc})")
            continue

        Tall = int(feats.shape[1])
        gt_latent = align_frame_gt_to_latent_frames(
            gt_raw, latent_len=Tall, frame_indices=frame_indices
        )
        if gt_latent.size < Tall:
            continue

        if predictor is None:
            predictor = make_predictor(
                feats,
                extras,
                args.ckpt,
                device,
                args.past_T,
                args.future_T,
            )

        windows = build_sliding_latent_windows(
            Tall, args.past_T, args.future_T, args.anomaly_stride
        )
        if not windows:
            continue

        with torch.no_grad():
            for (p0, p1), (f0, f1) in windows:
                if args.max_windows > 0 and len(labels) >= args.max_windows:
                    break
                pred, target = predict_window(
                    predictor, feats, extras, p0, p1, f0, f1, guidance_args
                )
                emb = window_embedding(pred, target, args.embedding)
                score, _ = compute_latent_anomaly_scores(
                    pred, target, args.anomaly_threshold
                )
                win_label = int(np.asarray(gt_latent[f0:f1]).max() > 0)
                embeddings.append(emb[0].detach().cpu().numpy().astype(np.float32))
                labels.append(win_label)
                scores.append(float(score[0].detach().cpu().item()))
                rows.append(
                    {
                        "cache_path": cache_path,
                        "video_path": video_path,
                        "window_p0": p0,
                        "window_p1": p1,
                        "window_f0": f0,
                        "window_f1": f1,
                        "label": win_label,
                        "score": scores[-1],
                    }
                )
        if (file_idx + 1) % 25 == 0:
            print(f"[INFO] processed files={file_idx + 1} windows={len(labels)}", flush=True)

    if len(labels) < 3:
        raise RuntimeError(f"Need at least 3 labeled windows for t-SNE, got {len(labels)}")
    if len(set(labels)) < 2:
        raise RuntimeError("Only one class found in collected windows; cannot inspect separation.")

    X = np.stack(embeddings, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    score_np = np.asarray(scores, dtype=np.float32)

    # Standardize before t-SNE so dimensions with large raw scale do not dominate.
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("Install scikit-learn first: pip install scikit-learn") from exc

    perplexity = min(float(args.perplexity), max(2.0, (len(y) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(args.seed),
    )
    points = tsne.fit_transform(X)

    np.savez_compressed(
        out_dir / "tsne_points.npz",
        points=points,
        labels=y,
        scores=score_np,
        embeddings=X,
    )
    with (out_dir / "windows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cache_path",
                "video_path",
                "window_p0",
                "window_p1",
                "window_f0",
                "window_f1",
                "label",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    png = out_dir / "tsne_error_embedding.png"
    build_plot(
        png,
        points=points,
        labels=y,
        scores=score_np,
        title=f"t-SNE of {args.embedding} windows (n={len(y)}, perplexity={perplexity:g})",
    )

    print(f"windows: {len(y)} normal={(y == 0).sum()} anomaly={(y == 1).sum()}")
    print(f"score_mean_normal: {score_np[y == 0].mean():.6f}")
    print(f"score_mean_anomaly: {score_np[y == 1].mean():.6f}")
    print(f"saved: {png}")
    print(f"saved: {out_dir / 'windows.csv'}")
    print(f"saved: {out_dir / 'tsne_points.npz'}")


if __name__ == "__main__":
    main()
