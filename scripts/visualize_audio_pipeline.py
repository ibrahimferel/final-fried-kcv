#!/usr/bin/env python3
"""
Generate pipeline visualizations: raw → resampled mono → linear STFT → Mel → 3ch tensor → Grad-CAM overlay.

Input: place .wav / .flac under project `samples/` (or pass --input).
Output: PNGs in `samples/` with prefix `pipeline_<stem>_`.

Usage (from `dsdba-fried-kcv/`):
  python scripts/visualize_audio_pipeline.py
  python scripts/visualize_audio_pipeline.py --input samples/my_clip.wav
  python scripts/visualize_audio_pipeline.py --demo
  python scripts/visualize_audio_pipeline.py --no-gradcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib import cm
from PIL import Image

from src.audio.dsp import (
    extract_mel_spectrogram,
    fix_duration,
    load_audio,
    normalise_spectrogram,
    resample_audio,
    to_mono,
    to_tensor,
    validate_duration,
)
from src.cv.model import DSDBAModel


def _load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _find_input_audio(samples_dir: Path) -> Path | None:
    for pat in ("*.wav", "*.flac"):
        for p in sorted(samples_dir.glob(pat)):
            if p.is_file() and not p.name.startswith("pipeline_"):
                return p
    return None


def _write_demo_wav(path: Path, cfg: dict) -> None:
    import soundfile as sf

    sr = int(cfg["audio"]["sample_rate"])
    n = int(cfg["audio"]["n_samples"])
    t = np.linspace(0.0, float(cfg["audio"]["duration_sec"]), num=n, endpoint=False, dtype=np.float32)
    y = 0.12 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    rng = np.random.default_rng(0)
    y = np.clip(y + 0.02 * rng.standard_normal(n).astype(np.float32), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def _save_waveform_1d(y: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    t = np.arange(y.shape[0], dtype=np.float64) / float(sr)
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=120)
    ax.plot(t, y, linewidth=0.6, color="#2563eb")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_raw_multichannel(waveform: np.ndarray, sr: int, out_path: Path) -> None:
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim == 1:
        _save_waveform_1d(wf, sr, "1) Raw audio (mono, before resample)", out_path)
        return
    ch = wf.shape[0]
    fig, axes = plt.subplots(ch, 1, figsize=(10, 2.2 * ch), sharex=True, dpi=120)
    if ch == 1:
        axes = [axes]
    t = np.arange(wf.shape[1], dtype=np.float64) / float(sr)
    for i, ax in enumerate(axes):
        ax.plot(t, wf[i], linewidth=0.6, color="#2563eb")
        ax.set_ylabel(f"Ch {i}")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("1) Raw audio (before resample / mono)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _linear_spectrogram_db(y: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    import librosa

    audio_cfg = cfg["audio"]
    sr = int(audio_cfg["sample_rate"])
    n_fft = int(audio_cfg["n_fft"])
    hop = int(audio_cfg["hop_length"])
    window = str(audio_cfg["window"])
    S = np.abs(
        librosa.stft(
            y.astype(np.float32, copy=False),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
        )
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, sr, hop


def _save_linear_spec(S_db: np.ndarray, sr: int, hop: int, out_path: Path) -> None:
    import librosa.display

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="linear",
        ax=ax,
    )
    ax.set_title("3) Linear-frequency spectrogram (STFT magnitude, dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_mel_spec(mel_power: np.ndarray, cfg: dict, out_path: Path) -> None:
    import librosa
    import librosa.display

    audio_cfg = cfg["audio"]
    sr = int(audio_cfg["sample_rate"])
    hop = int(audio_cfg["hop_length"])
    mel_db = librosa.power_to_db(mel_power.astype(np.float32, copy=False), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="mel",
        fmax=float(cfg["gradcam"]["mel_fmax_hz"]),
        ax=ax,
    )
    ax.set_title("4) Mel spectrogram (power → dB, before resize / 3ch duplicate)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_tensor_duplicate_preview(tensor, out_path: Path) -> None:
    x = tensor.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    identical = bool(
        np.allclose(x[0].numpy(), x[1].numpy()) and np.allclose(x[1].numpy(), x[2].numpy())
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=120)
    for i, ax in enumerate(axes):
        ax.imshow(x[i].numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"Channel {i}")
        ax.axis("off")
    fig.suptitle(
        "5) After resize + 3-channel duplicate (EfficientNet input)"
        + (" — channels identical" if identical else ""),
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_gradcam_overlay(tensor, saliency: np.ndarray, cfg: dict, out_path: Path) -> None:
    grad_cfg = cfg["gradcam"]
    alpha = float(grad_cfg["overlay_alpha"])
    cmap_name = str(grad_cfg["colormap"])
    x = tensor[0] if tensor.ndim == 4 else tensor
    image = x.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    colormap = cm.get_cmap(cmap_name)
    heatmap = colormap(np.clip(saliency, 0.0, 1.0))[..., :3].astype(np.float32)
    overlay = np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)
    overlay_u8 = (overlay * 255.0).astype(np.uint8)
    Image.fromarray(overlay_u8).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DSDBA audio → tensor → Grad-CAM pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Path to .wav or .flac")
    parser.add_argument("--demo", action="store_true", help="Write samples/demo_pipeline.wav if no input")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip step 6 (Grad-CAM)")
    args = parser.parse_args()

    cfg = _load_cfg()
    samples_dir = ROOT / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    in_path = args.input
    if in_path is None:
        in_path = _find_input_audio(samples_dir)
    if in_path is None and args.demo:
        demo_path = samples_dir / "demo_pipeline.wav"
        _write_demo_wav(demo_path, cfg)
        in_path = demo_path
    if in_path is None:
        print("No audio in samples/. Add .wav/.flac or use --demo.", file=sys.stderr)
        return 1

    in_path = in_path.resolve()
    if not in_path.is_file():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    stem = in_path.stem

    waveform, sample_rate = load_audio(in_path, cfg)
    validate_duration(waveform=waveform, sample_rate=sample_rate, cfg=cfg)

    _save_raw_multichannel(waveform, sample_rate, samples_dir / f"pipeline_{stem}_01_raw.png")

    resampled = resample_audio(waveform=waveform, orig_sr=sample_rate, cfg=cfg)
    mono_rs = to_mono(resampled)
    _save_waveform_1d(
        mono_rs,
        int(cfg["audio"]["sample_rate"]),
        "2) After resample + mono (before fixed-duration crop/pad)",
        samples_dir / f"pipeline_{stem}_02_resampled_mono.png",
    )

    fixed = fix_duration(mono_rs, cfg=cfg)
    S_db, sr, hop = _linear_spectrogram_db(fixed, cfg)
    _save_linear_spec(S_db, sr, hop, samples_dir / f"pipeline_{stem}_03_linear_spec.png")

    mel = extract_mel_spectrogram(fixed, cfg=cfg)
    _save_mel_spec(mel, cfg, samples_dir / f"pipeline_{stem}_04_mel_spec.png")

    normalised = normalise_spectrogram(mel)
    tensor = to_tensor(normalised, cfg=cfg)
    _save_tensor_duplicate_preview(tensor, samples_dir / f"pipeline_{stem}_05_tensor_3ch.png")

    if not args.no_gradcam:
        try:
            from src.cv.gradcam import compute_gradcam
        except ImportError as exc:
            print(
                "Grad-CAM skipped (missing dependency). Install grad-cam / torch stack. "
                f"Reason: {exc}",
                file=sys.stderr,
            )
        else:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DSDBAModel(cfg=cfg, pretrained=True).to(device)
            ckpt = ROOT / "models" / "checkpoints" / str(
                cfg.get("training", {}).get("best_checkpoint_filename", "best_model.pth")
            )
            if ckpt.is_file():
                payload = torch.load(str(ckpt), map_location=device)
                state = payload.get("model_state_dict", payload)
                model.load_state_dict(state, strict=False)
            model.eval()
            t = tensor.to(device=device, dtype=torch.float32)
            saliency = compute_gradcam(model=model, tensor=t, cfg=cfg)
            _save_gradcam_overlay(
                t,
                saliency,
                cfg,
                samples_dir / f"pipeline_{stem}_06_gradcam_overlay.png",
            )

    print(f"Wrote images under {samples_dir} (stem={stem})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
