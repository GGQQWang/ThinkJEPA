#!/usr/bin/env python3
# ThinkJEPA utility: append V-JEPA features to existing Qwen/VLM cache archives.

import argparse
import glob
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA2_ROOT = Path(
    os.environ.get("VJEPA2_ROOT", str(REPO_ROOT / "vjepa2"))
).resolve()
for _path in (
    REPO_ROOT,
    REPO_ROOT / "cache_train",
    VJEPA2_ROOT,
    VJEPA2_ROOT.parent,
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from cache_train.checkpoint_paths import resolve_dense_jepa_checkpoint  # noqa: E402
from vjepa2.src.models.vision_transformer import vit_large_rope  # noqa: E402


_QWEN_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>.+?)_L\d+_nf\d+_res\d+_new\d+_s\d+of\d+$"
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(
        "Append V-JEPA dense features to existing Qwen3-VL .npz caches."
    )
    p.add_argument("--video_manifest", required=True, help="Text file with mp4 paths.")
    p.add_argument("--video_root", required=True, help="Root used when Qwen cache was created.")
    p.add_argument("--cache_dir", required=True, help="Existing Qwen cache directory.")
    p.add_argument("--num_frames", type=int, default=48, help="Uniform frames to sample.")
    p.add_argument("--decode_size", type=int, default=256, help="Decode resize size.")
    p.add_argument("--save_dtype", choices=["fp16", "fp32", "bf16"], default="fp16")
    p.add_argument("--save_mode", choices=["compressed", "raw"], default="compressed")
    p.add_argument("--overwrite", action="store_true", help="Recompute existing vjepa_feats.")
    p.add_argument("--limit", type=int, default=0, help="Optional max videos for smoke tests.")
    return p.parse_args()


def normalize_qwen_cache_stem(npz_path: str) -> str:
    base = os.path.splitext(os.path.basename(npz_path))[0]
    m = _QWEN_CACHE_NAME_RE.match(base)
    return m.group("stem") if m else base


def make_safe_video_identifier(p: str) -> str:
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def relative_video_subdirectory(video_path: str, dataset_root: str) -> str:
    try:
        rel = os.path.relpath(
            os.path.normpath(video_path), start=os.path.normpath(dataset_root)
        )
        subdir = os.path.dirname(rel)
        if subdir.startswith(".."):
            return ""
        return subdir
    except Exception:
        return ""


def load_manifest(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        out = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    return [x for x in out if x.lower().endswith(".mp4")]


def build_cache_index(cache_dir: str):
    by_rel_and_stem = {}
    by_stem = {}
    for p in sorted(glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True)):
        rel = os.path.relpath(p, cache_dir)
        rel_dir = os.path.normpath(os.path.dirname(rel))
        stem = normalize_qwen_cache_stem(p)
        by_rel_and_stem[(rel_dir, stem)] = p
        by_stem.setdefault(stem, []).append(p)
    return by_rel_and_stem, by_stem


def resolve_cache_path(video_path: str, video_root: str, cache_dir: str, by_rel_and_stem, by_stem):
    stem = make_safe_video_identifier(video_path)
    rel_dir = os.path.normpath(relative_video_subdirectory(video_path, video_root))
    rel_key = (rel_dir, stem)
    if rel_key in by_rel_and_stem:
        return by_rel_and_stem[rel_key]
    if stem in by_stem and len(by_stem[stem]) == 1:
        return by_stem[stem][0]
    # Last-resort direct glob for users who moved cache files after extraction.
    candidates = sorted(glob.glob(os.path.join(cache_dir, "**", f"{stem}_*.npz"), recursive=True))
    return candidates[0] if candidates else None


def uniform_frame_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        raise ValueError("empty video")
    if num_frames <= 0:
        raise ValueError("--num_frames must be positive")
    return np.linspace(0, total_frames - 1, num=int(num_frames), dtype=np.int64)


def read_video_clip(video_path: str, num_frames: int, decode_size: int):
    vr = VideoReader(video_path, ctx=cpu(0), width=int(decode_size), height=int(decode_size))
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"Empty video: {video_path}")
    idx = uniform_frame_indices(total, num_frames)
    frames = vr.get_batch(idx).asnumpy()
    return frames.astype(np.uint8, copy=False), total


def load_dense_jepa_encoder(pt_model_path=None):
    if pt_model_path is None:
        pt_model_path = resolve_dense_jepa_checkpoint()
    model_pt = vit_large_rope(img_size=(256, 256), num_frames=64)
    model_pt.cuda().eval()
    checkpoint = torch.load(pt_model_path, weights_only=True, map_location="cpu")
    state = checkpoint["encoder"] if isinstance(checkpoint, dict) and "encoder" in checkpoint else checkpoint
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model_pt.load_state_dict(state, strict=False)
    print(f"[INFO] V-JEPA weights loaded from {pt_model_path}: {msg}", flush=True)
    return model_pt


def preprocess_vjepa_video(frames: np.ndarray):
    video = torch.from_numpy(frames).cuda(non_blocking=True)
    if video.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C] frames, got {tuple(video.shape)}")
    if video.shape[-1] == 4:
        video = video[..., :3]
    video = video.float()
    if video.max() > 1.0:
        video = video / 255.0
    # [T,H,W,C] -> [1,T,C,H,W]
    video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    B, T, C, H, W = video.shape
    frames_flat = video.view(B * T, C, H, W)
    short_side = int(256.0 / 224 * 256)
    if H <= W:
        new_h = short_side
        new_w = int(round(W * (short_side / H)))
    else:
        new_h = int(round(H * (short_side / W)))
        new_w = short_side
    frames_flat = F.interpolate(
        frames_flat,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    top = max(0, int(round((new_h - 256) / 2.0)))
    left = max(0, int(round((new_w - 256) / 2.0)))
    frames_flat = frames_flat[:, :, top : top + 256, left : left + 256]
    mean = torch.tensor(_IMAGENET_MEAN, device=frames_flat.device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(_IMAGENET_STD, device=frames_flat.device, dtype=torch.float32)[None, :, None, None]
    frames_flat = (frames_flat - mean) / std
    return frames_flat.view(B, T, C, 256, 256).contiguous()


def encode_dense_jepa_video(video, model_pt):
    with torch.no_grad():
        B, T, C, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4)
        out = model_pt(video)
        out = out.contiguous().view(B, T, -1, out.shape[-1])
    return out


def to_cache_dtype(x: torch.Tensor, save_dtype: str):
    if save_dtype == "fp16":
        return x.detach().float().half().cpu().numpy()
    if save_dtype == "bf16":
        return x.detach().bfloat16().cpu().view(torch.uint16).numpy()
    return x.detach().float().cpu().numpy()


def dummy_geometry(num_frames: int):
    xyz = np.zeros((num_frames, 1, 3), dtype=np.float32)
    rot = np.broadcast_to(np.eye(3, dtype=np.float32), (num_frames, 1, 3, 3)).copy()
    tf = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 1, 4, 4)).copy()
    cam_ext = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 4, 4)).copy()
    cam_int_single = np.array(
        [[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cam_int = np.broadcast_to(cam_int_single, (num_frames, 3, 3)).copy()
    confs = np.ones((num_frames, 1), dtype=np.float32)
    return {
        "xyz_cam": xyz,
        "R_cam": rot,
        "xyz_world": xyz.copy(),
        "R_world": rot.copy(),
        "tfs_in_cam": tf,
        "tfs": tf.copy(),
        "cam_ext": cam_ext,
        "cam_int": cam_int,
        "confs": confs,
        "lang_instruct": np.asarray(""),
    }


def read_npz_copy(path: str):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def write_npz_atomic(path: str, payload: dict, save_mode: str):
    tmp = f"{path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npz"
    if save_mode == "raw":
        np.savez(tmp, **payload)
    else:
        np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def main():
    args = parse_args()
    videos = load_manifest(args.video_manifest)
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise ValueError(f"No mp4 paths found in manifest: {args.video_manifest}")

    by_rel_and_stem, by_stem = build_cache_index(args.cache_dir)
    if not by_rel_and_stem:
        raise ValueError(f"No .npz cache files found under: {args.cache_dir}")

    model_pt = load_dense_jepa_encoder()
    for p in model_pt.parameters():
        p.requires_grad_(False)

    done = skipped = missing = failed = 0
    for i, video_path in enumerate(videos, start=1):
        cache_path = resolve_cache_path(
            video_path,
            args.video_root,
            args.cache_dir,
            by_rel_and_stem,
            by_stem,
        )
        if cache_path is None:
            missing += 1
            print(f"[MISS] cache not found for video={video_path}", flush=True)
            continue
        try:
            payload = read_npz_copy(cache_path)
            if (not args.overwrite) and "vjepa_feats" in payload:
                skipped += 1
                continue

            frames, total_frames = read_video_clip(
                video_path, num_frames=args.num_frames, decode_size=args.decode_size
            )
            video = preprocess_vjepa_video(frames)
            with torch.no_grad():
                feats = encode_dense_jepa_video(video, model_pt)[0].contiguous()

            payload["imgs"] = frames
            payload["vjepa_feats"] = to_cache_dtype(feats, args.save_dtype)
            payload["path"] = np.asarray(str(video_path))
            payload["video_path"] = np.asarray(str(video_path))
            payload["nframes_used"] = np.asarray(int(args.num_frames), dtype=np.int32)
            payload["total_frames"] = np.asarray(int(total_frames), dtype=np.int32)
            payload.update(dummy_geometry(int(args.num_frames)))
            write_npz_atomic(cache_path, payload, save_mode=args.save_mode)
            done += 1
            if done == 1 or done % 25 == 0:
                print(
                    f"[OK] {done} updated | {i}/{len(videos)} | {cache_path} "
                    f"vjepa_feats={tuple(payload['vjepa_feats'].shape)}",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {video_path}: {type(exc).__name__}: {exc}", flush=True)

    print(
        f"[DONE] updated={done} skipped={skipped} missing_cache={missing} failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
