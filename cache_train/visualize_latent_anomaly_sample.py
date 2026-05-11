import argparse
import csv
import glob
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cache_train.thinker_predictor import CortexGuidedVideoPredictor
from cache_train.thinker_train import (
    align_frame_gt_to_latent_frames,
    build_frame_gt_index,
    build_sliding_latent_windows,
    build_thinkjepa_guidance_inputs,
    compute_latent_anomaly_scores,
    load_frame_gt_array,
    _normalize_thinker_cache_stem,
)


def parse_args():
    p = argparse.ArgumentParser("Visualize one latent-anomaly sample.")
    p.add_argument("--cache_path", default="", help="Specific cached .npz sample.")
    p.add_argument("--cache_dir", default="", help="Cache root used when selecting by --index/--stem.")
    p.add_argument("--index", type=int, default=0, help="Sample index in cache_dir when cache_path is omitted.")
    p.add_argument("--stem", default="", help="Pick the first cache file whose normalized stem matches this.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path, e.g. ckpt_latest.pt.")
    p.add_argument("--frame_gt_dir", default="", help="Directory containing per-frame 0/1 GT labels.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--past_T", type=int, default=32)
    p.add_argument("--future_T", type=int, default=16)
    p.add_argument("--anomaly_stride", type=int, default=16)
    p.add_argument("--anomaly_threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve_cache_path(args):
    if args.cache_path:
        return args.cache_path
    files = sorted(glob.glob(os.path.join(args.cache_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .npz files under {args.cache_dir}")
    if args.stem:
        for path in files:
            if _normalize_thinker_cache_stem(path) == args.stem:
                return path
        raise FileNotFoundError(f"No cache file matched stem={args.stem}")
    return files[int(args.index)]


def to_tensor(x, device, dtype=torch.float32):
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def load_npz_sample(path, device):
    with np.load(path, allow_pickle=False) as z:
        feats = to_tensor(z["vjepa_feats"], device).unsqueeze(0)
        extras = {}
        for key in ("vlm_old", "vlm_new"):
            if key in z:
                x = to_tensor(z[key], device)
                if x.dim() == 4:
                    x = x[:, -1, :, :]
                extras[key] = x
        if "token_ids" in z:
            extras["token_ids"] = torch.from_numpy(z["token_ids"]).to(device=device)
        if "layers" in z:
            extras["layers"] = torch.from_numpy(z["layers"]).to(device=device)
        frame_indices = None
        if "frame_indices" in z:
            frame_indices = np.asarray(z["frame_indices"], dtype=np.int64).reshape(-1)
        elif "total_frames" in z:
            total_frames = int(np.asarray(z["total_frames"]).reshape(-1)[0])
            frame_indices = np.linspace(0, total_frames - 1, num=feats.shape[1], dtype=np.int64)
        video_path = str(np.asarray(z["video_path"]).item()) if "video_path" in z else ""
    return feats, extras, frame_indices, video_path


def make_predictor(feats, extras, ckpt_path, device, past_T, future_T):
    _, Tall, P, D = feats.shape
    total_frames = min(Tall, int(past_T) + int(future_T))
    old_dim = int(extras["vlm_old"].shape[-1]) if "vlm_old" in extras else 1
    new_dim = int(extras["vlm_new"].shape[-1]) if "vlm_new" in extras else 1
    predictor = CortexGuidedVideoPredictor(
        img_size=(P, 1),
        patch_size=1,
        num_frames=total_frames,
        tubelet_size=1,
        embed_dim=D,
        predictor_embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=True,
        use_vlm_merge=True,
        vlm_cond_mode="film",
        vlm_old_dim=old_dim,
        vlm_new_dim=new_dim,
    ).to(device)
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob.get("predictor", blob)
    predictor.load_state_dict(state, strict=True)
    predictor.eval()
    return predictor


def flatten_temporal_patch_tokens(feats):
    B, T, P, D = feats.shape
    return feats.reshape(B, T * P, D)


def build_temporal_patch_indices(patches_per_frame, start, end):
    ids = []
    for t in range(int(start), int(end)):
        base = t * int(patches_per_frame)
        ids.extend(range(base, base + int(patches_per_frame)))
    return torch.tensor(ids, dtype=torch.long)


def predict_window(predictor, feats, extras, p0, p1, f0, f1, args):
    B, Tall, P, D = feats.shape
    total = (p1 - p0) + (f1 - f0)
    feats_total = feats[:, p0:f1].contiguous()
    x_seq = flatten_temporal_patch_tokens(feats_total)
    ctx_len = p1 - p0
    tgt_len = f1 - f0
    idx_ctx = build_temporal_patch_indices(P, 0, ctx_len).to(feats.device)
    idx_tgt = build_temporal_patch_indices(P, ctx_len, ctx_len + tgt_len).to(feats.device)
    masks_x = idx_ctx.unsqueeze(0).expand(B, -1)
    masks_y = idx_tgt.unsqueeze(0).expand(B, -1)
    x_ctxt = x_seq.gather(dim=1, index=masks_x.unsqueeze(-1).expand(-1, -1, D))
    ext = build_thinkjepa_guidance_inputs(extras=extras, args=args, device=feats.device)
    pred_seq = predictor(x_ctxt, masks_x, masks_y, ext=ext)
    pred = pred_seq.view(B, tgt_len, P, D)
    target = feats[:, f0:f1].detach()
    return pred, target


def load_gt_for_cache(cache_path, gt_dir):
    if not gt_dir:
        return None
    index = build_frame_gt_index(gt_dir)
    stem = _normalize_thinker_cache_stem(cache_path)
    label_path = index.get(stem)
    if label_path is None:
        return None
    return load_frame_gt_array(label_path)


def write_plot(out_png, scores, labels=None, threshold=None, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(12, 4))
    if labels is not None and len(labels) == len(scores):
        ax.fill_between(x, 0, 1, where=labels > 0, transform=ax.get_xaxis_transform(), color="tab:red", alpha=0.18, label="GT anomaly")
    ax.plot(x, scores, color="tab:blue", linewidth=1.8, label="anomaly score")
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.2, label=f"threshold={threshold:g}")
    ax.set_xlabel("sampled latent frame")
    ax.set_ylabel("prediction error")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cache_path = resolve_cache_path(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, extras, frame_indices, video_path = load_npz_sample(cache_path, device)
    predictor = make_predictor(feats, extras, args.ckpt, device, args.past_T, args.future_T)

    Tall = int(feats.shape[1])
    score_sum = torch.zeros((Tall,), device=device)
    score_count = torch.zeros((Tall,), device=device)
    windows = build_sliding_latent_windows(Tall, args.past_T, args.future_T, args.anomaly_stride)
    with torch.no_grad():
        for p0, p1, f0, f1 in windows:
            pred, target = predict_window(predictor, feats, extras, p0, p1, f0, f1, args)
            scores, _ = compute_latent_anomaly_scores(pred, target, args.anomaly_threshold)
            score_sum[f0:f1] += scores[0]
            score_count[f0:f1] += 1.0

    frame_scores = (score_sum / score_count.clamp_min(1.0)).detach().cpu().numpy()
    valid = (score_count > 0).detach().cpu().numpy().astype(bool)

    gt_raw = load_gt_for_cache(cache_path, args.frame_gt_dir)
    labels = None
    if gt_raw is not None:
        labels = align_frame_gt_to_latent_frames(gt_raw, latent_len=Tall, frame_indices=frame_indices)

    csv_path = out_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["latent_frame", "source_frame", "score", "valid", "gt"])
        for i in range(Tall):
            src = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
            gt = int(labels[i]) if labels is not None and i < len(labels) else ""
            writer.writerow([i, src, float(frame_scores[i]), int(valid[i]), gt])

    png_path = out_dir / "score_vs_gt.png"
    write_plot(
        png_path,
        frame_scores,
        labels=labels,
        threshold=args.anomaly_threshold,
        title=os.path.basename(video_path or cache_path),
    )
    print(f"cache_path: {cache_path}")
    print(f"video_path: {video_path}")
    print(f"windows: {len(windows)}")
    print(f"saved: {png_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
