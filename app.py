"""
Module: app
SRS Reference: FR-DEP-001–010
SDLC Phase: 4 - Implementation (Sprint E)
Sprint: E
Pipeline Stage: Deployment
Purpose: Gradio 4.x UI wiring for the full DSDBA pipeline with non-blocking NLP explanation.
Dependencies: gradio, asyncio, PyYAML, torch, onnxruntime, matplotlib, pydub
Interface Contract:
  Input: audio filepath (<= 20 MB) from Gradio upload or microphone; normalized to WAV before DSP
  Output: UI components showing CV verdict+confidence, Grad-CAM overlay, band chart, and NLP explanation (with fallback badge)
Latency Target: <= 15,000 ms end-to-end CPU per FR-DEP-007
Open Questions Resolved: Q6 (Gradio 4.x locked)
Open Questions Blocking: None for Sprint E UI wiring
MCP Tools Used: context7-mcp (Gradio API), huggingface-mcp (Spaces guidance)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-04-02
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import numpy as np
import pandas as pd
import torch
import yaml

from src.audio.dsp import preprocess_audio
from src.cv.gradcam import run_gradcam
from src.cv.infer import export_to_onnx, load_onnx_session, run_onnx_inference
from src.cv.model import DSDBAModel
from src.nlp.explain import generate_explanation
from src.utils.errors import DSDBAError
from src.utils.logger import log_error, log_info, log_warning


# ── Config & helpers ───────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("Failed to parse config.yaml") from exc


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _coerce_audio_filepath(raw: Any) -> str | None:
    """
    Normalize Gradio Audio payloads to a filesystem path string.

    Spaces / older clients may send ``FileData``-like dicts (``{"path": "..."}``)
    instead of a bare string; examples usually pass a plain path.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        path = raw.strip()
        return path if path else None
    if isinstance(raw, Path):
        return str(raw)
    path_key = getattr(raw, "path", None)
    if isinstance(path_key, str) and path_key.strip():
        return path_key.strip()
    if isinstance(raw, dict):
        for key in ("path", "name", "file_name"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _models_dir() -> Path:
    return _project_root() / "models" / "checkpoints"


# ── Audio conversion (MP3 / browser record / etc. → PCM WAV) ─────────────────

def _is_riff_wave_file(path: Path) -> bool:
    """True only for a classic RIFF WAVE header (PCM WAV torchaudio can load)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return False
    return len(head) == 12 and head[0:4] == b"RIFF" and head[8:12] == b"WAVE"


def _pydub_format_for_path(path: Path) -> str | None:
    """ffmpeg format string for known extensions; None lets ffmpeg probe the file."""
    ext = path.suffix.lower().lstrip(".")
    return {
        "mp3": "mp3",
        "mp4": "mp4",
        "m4a": "m4a",
        "ogg": "ogg",
        "oga": "ogg",
        "opus": "opus",
        "webm": "webm",
        "flac": "flac",
    }.get(ext)


def _convert_to_wav(input_path: Path) -> Path:
    """
    Normalize uploads and microphone captures to a real PCM WAV on disk.

    Gradio often saves recordings with a ``.wav`` name while the payload is MP3
    (or WebM). Those must be re-encoded via ffmpeg/pydub before ``torchaudio``.
    """
    suffix = input_path.suffix.lower()
    if suffix == ".wav" and _is_riff_wave_file(input_path):
        return input_path

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise DSDBAError(
            "AUD-002", "pydub is required for compressed / recorded audio.", "deployment"
        ) from exc

    out = Path(tempfile.gettempdir()) / f"dsdba_converted_{int(time.time() * 1000)}.wav"
    fmt = _pydub_format_for_path(input_path)
    try:
        if fmt is not None:
            audio = AudioSegment.from_file(str(input_path), format=fmt)
        else:
            audio = AudioSegment.from_file(str(input_path))
        audio.export(str(out), format="wav")
        log_info(
            stage="deployment",
            message="audio_converted_to_wav",
            data={"from_suffix": suffix or "(none)", "pydub_format": fmt, "to": str(out)},
        )
        return out
    except Exception as exc:
        log_warning(stage="deployment", message="audio_conversion_failed",
                    data={"reason": str(exc)})
        raise DSDBAError(
            "AUD-002", f"Cannot convert {suffix or 'audio'} to WAV: {exc}", "deployment"
        ) from exc


# ── Model & ONNX loading ───────────────────────────────────────────────────────

def _ensure_onnx_session(cfg: dict[str, Any], model: DSDBAModel) -> Any:
    """Load ONNX session from HuggingFace Hub, fallback to local, fallback to export."""
    from huggingface_hub import hf_hub_download

    try:
        onnx_path = hf_hub_download(
            repo_id="narcissablack/fake67",
            filename="dsdba_efficientnet_b4.onnx"
        )
        log_info(stage="deployment", message="onnx_loaded_from_hub",
                 data={"repo": "narcissablack/fake67"})
    except Exception:
        local_path = _models_dir() / "dsdba_efficientnet_b4.onnx"
        if local_path.exists():
            onnx_path = str(local_path)
            log_warning(stage="deployment", message="onnx_loaded_from_local",
                        data={"path": str(local_path)})
        else:
            log_warning(stage="deployment", message="onnx_missing_exporting",
                        data={"path": str(local_path)})
            onnx_path = export_to_onnx(model=model, cfg=cfg)

    return load_onnx_session(onnx_path=onnx_path, cfg=cfg)


def _maybe_load_weights(model: DSDBAModel, cfg: dict[str, Any]) -> None:
    """Load checkpoint from HF Hub first, fallback to local."""
    from huggingface_hub import hf_hub_download

    ckpt_name = str(cfg.get("training", {}).get("best_checkpoint_filename", "best_model.pth"))
    ckpt_path = None

    try:
        ckpt_path = hf_hub_download(
            repo_id="narcissablack/fake67",
            filename=ckpt_name
        )
        log_info(stage="deployment", message="checkpoint_loaded_from_hub",
                 data={"repo": "narcissablack/fake67", "filename": ckpt_name})
    except Exception:
        local_path = _models_dir() / ckpt_name
        if local_path.exists():
            ckpt_path = str(local_path)
            log_warning(stage="deployment", message="checkpoint_loaded_from_local",
                        data={"path": str(local_path)})
        else:
            log_warning(stage="deployment", message="checkpoint_missing_random_weights",
                        data={"path": str(local_path)})
            return

    try:
        payload = torch.load(str(ckpt_path), map_location="cpu")
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=False)
        log_info(stage="deployment", message="checkpoint_loaded",
                 data={"path": str(ckpt_path)})
    except Exception as exc:
        log_warning(stage="deployment", message="checkpoint_load_failed_random_weights",
                    data={"path": str(ckpt_path), "reason": str(exc)})


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _validate_file_size(audio_path: Path, cfg: dict[str, Any]) -> None:
    max_mb = float(cfg["deployment"]["max_upload_mb"])
    size_bytes = int(audio_path.stat().st_size)
    if size_bytes > int(max_mb * 1024 * 1024):
        raise ValueError("FILE_TOO_LARGE")


def _band_df(band_pct: dict[str, float]) -> pd.DataFrame:
    order = ["low", "low_mid", "high_mid", "high"]
    bands = [b for b in order if b in band_pct]
    perc = [float(band_pct[b]) for b in bands]
    return pd.DataFrame({"band": bands, "percent": perc})


def _confidence_percent(conf: float) -> float:
    return float(conf) * 100.0


def _verdict_html(label: str, confidence: float) -> str:
    pct = max(0.0, min(100.0, _confidence_percent(confidence)))
    color = "#ef4444" if str(label).lower() == "spoof" else "#22c55e"
    return (
        "<div style='width: 100%; background: #e5e7eb; border-radius: 8px; overflow: hidden;'>"
        f"<div style='width: {pct:.2f}%; background: {color}; padding: 6px 0; "
        f"color: white; text-align: center; font-weight: 600;'>"
        f"{pct:.2f}%"
        "</div></div>"
    )


def _spectrogram_image_from_tensor(tensor: torch.Tensor) -> Path:
    import matplotlib.pyplot as plt

    x = tensor.detach().cpu()
    img = x[0].numpy().astype(np.float32, copy=False)
    out = Path(tempfile.gettempdir()) / f"dsdba_spec_{int(time.time() * 1000)}.png"
    plt.figure(figsize=(5, 4), dpi=120)
    plt.imshow(img, aspect="auto", origin="lower", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out


def ensure_demo_samples(cfg: dict[str, Any]) -> list[Path]:
    import soundfile as sf

    root = _project_root() / "data" / "samples"
    root.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["audio"]["sample_rate"])
    n = int(cfg["audio"]["n_samples"])
    t = np.linspace(0.0, float(cfg["audio"]["duration_sec"]), num=n, endpoint=False, dtype=np.float32)

    samples: list[tuple[str, np.ndarray]] = [
        ("bonafide_01.wav", 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)),
        ("bonafide_02.wav", 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)),
    ]
    rng = np.random.default_rng(0)
    noise = (0.03 * rng.standard_normal(size=n)).astype(np.float32)
    hf = (0.06 * np.sin(2.0 * np.pi * 3200.0 * t)).astype(np.float32)
    samples.append(("spoof_01.wav", np.clip(noise + hf, -1.0, 1.0)))
    samples.append(("spoof_02.wav", np.clip(
        noise + 0.06 * np.sin(2.0 * np.pi * 5200.0 * t).astype(np.float32), -1.0, 1.0
    )))

    paths: list[Path] = []
    for name, wav in samples:
        p = root / name
        if not p.exists():
            sf.write(str(p), wav, sr, subtype="PCM_16")
        paths.append(p)
    return paths


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    audio_file: str | Path,
    cfg: dict[str, Any],
    onnx_session: Any,
    model: DSDBAModel,
):
    audio_path = Path(audio_file)
    _validate_file_size(audio_path=audio_path, cfg=cfg)

    # ✅ Microphone / MP3 / mislabeled .wav → PCM WAV before DSP (torchaudio)
    audio_path = _convert_to_wav(audio_path)

    tensor = preprocess_audio(file_path=audio_path, cfg=cfg)
    label, confidence = run_onnx_inference(session=onnx_session, tensor=tensor, cfg=cfg)
    heatmap_path, band_pct = run_gradcam(tensor=tensor, model=model, cfg=cfg)

    explanation_task = asyncio.create_task(
        generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg)
    )
    spec_path = _spectrogram_image_from_tensor(tensor)
    return label, confidence, spec_path, heatmap_path, band_pct, explanation_task, audio_path


# ── Startup: load config + model + ONNX session ONCE ──────────────────────────

CFG: dict[str, Any] = load_config("config.yaml")
MODEL: DSDBAModel = DSDBAModel(cfg=CFG, pretrained=False)
_maybe_load_weights(model=MODEL, cfg=CFG)
MODEL.eval()
ONNX_SESSION = _ensure_onnx_session(cfg=CFG, model=MODEL)


# ── UI callbacks ───────────────────────────────────────────────────────────────

def _ui_error_outputs() -> tuple:
    return (None, None, "", None, None, None, None, "")


async def ui_run(audio_path: Any):
    path_str = _coerce_audio_filepath(audio_path)
    if not path_str:
        gr.Warning("Please upload or record an audio clip.")
        yield _ui_error_outputs()
        return

    start = time.perf_counter()
    try:
        (
            label,
            confidence,
            spec_path,
            heatmap_path,
            band_pct,
            explanation_task,
            wav_for_display,
        ) = run_pipeline(
            audio_file=path_str,
            cfg=CFG,
            onnx_session=ONNX_SESSION,
            model=MODEL,
        )

        # Yield CV results dulu (tidak nunggu NLP)
        yield (
            label,
            _confidence_percent(confidence),
            _verdict_html(label=label, confidence=confidence),
            str(wav_for_display),
            str(spec_path),
            str(heatmap_path),
            _band_df(band_pct),
            "⏳ Generating explanation…",
        )

        # Tunggu NLP selesai
        explanation_text, api_was_used = await explanation_task
        if not api_was_used:
            gr.Warning(str(CFG["nlp"].get("warning_badge_text", "AI explanation unavailable — rule-based fallback used.")))

        yield (
            label,
            _confidence_percent(confidence),
            _verdict_html(label=label, confidence=confidence),
            str(wav_for_display),
            str(spec_path),
            str(heatmap_path),
            _band_df(band_pct),
            explanation_text,
        )

    except ValueError as exc:
        if str(exc) == "FILE_TOO_LARGE":
            gr.Warning("File too large (> 20 MB). Please upload a smaller file.")
        else:
            gr.Error("Failed to process input.")
            log_error(stage="deployment", message="ui_value_error", data={"reason": str(exc)})
        yield _ui_error_outputs()

    except DSDBAError as exc:
        if exc.code == "AUD-001":
            gr.Warning("Audio too short (< 0.5 s). Please upload a longer clip.")
        elif exc.code == "AUD-002":
            gr.Warning("Unsupported format. Please upload WAV, FLAC, MP3, or MP4.")
        else:
            gr.Warning("Audio processing failed.")
        yield _ui_error_outputs()

    except Exception as exc:
        gr.Error("Unexpected error. Please try again.")
        log_error(stage="deployment", message="ui_exception", data={"reason": str(exc)})
        yield _ui_error_outputs()

    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log_info(stage="deployment", message="ui_run_complete",
                 data={"latency_ms": round(elapsed_ms, 3)})


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    demo_samples = ensure_demo_samples(CFG)

    with gr.Blocks(title="DSDBA — Deepfake Speech Detection") as demo:
        gr.Markdown("## DSDBA — Deepfake Speech Detection & Explainability")
        gr.Markdown(
            "Upload or record audio (≤ 20 MB). Recordings are converted to WAV internally "
            "(including browser MP3/WebM). CV results appear first; explanation loads asynchronously."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Gradio 4.36.x (default on many Spaces) does not support `file_types` on Audio;
                # conversion in `_convert_to_wav` still accepts MP3/WebM/mislabeled WAV from upload or mic.
                audio_in = gr.Audio(
                    label="Upload or record",
                    type="filepath",
                    format="wav",
                    sources=["upload", "microphone"],
                )
                run_btn = gr.Button("Run", variant="primary")
                gr.Examples(
                    examples=[[str(p)] for p in demo_samples],
                    inputs=[audio_in],
                    label="Demo examples (synthetic tones/noise)",
                )

            with gr.Column(scale=2):
                with gr.Row():
                    verdict = gr.Label(label="Verdict")
                    confidence_pct = gr.Number(label="Confidence (%)", precision=2)
                conf_bar = gr.HTML(label="Confidence bar")

                waveform = gr.Audio(label="Waveform", type="filepath")
                spec_img = gr.Image(label="Spectrogram", type="filepath")

                gradcam_img = gr.Image(label="Grad-CAM overlay", type="filepath")
                band_plot = gr.BarPlot(
                    label="Band attribution (%)",
                    x="band",
                    y="percent",
                )

                gr.Markdown("**AI-generated explanation (English)**")
                explanation = gr.Textbox(label="Explanation", lines=6)

        run_btn.click(
            fn=ui_run,
            inputs=[audio_in],
            outputs=[verdict, confidence_pct, conf_bar, waveform, spec_img,
                     gradcam_img, band_plot, explanation],
        )

        with gr.Accordion("About", open=False):
            gr.Markdown(
                "\n".join([
                    "### About DSDBA",
                    "**Pipeline**: Audio DSP → EfficientNet-B4 (ONNX) → Grad-CAM → LLM explanation (rule-based fallback).",
                    "",
                    "**Supported formats**: WAV, FLAC, MP3, MP4, M4A, OGG, WebM; microphone capture is converted to WAV via ffmpeg/pydub.",
                    "",
                    "**Dataset citation**: Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.",
                    "",
                    "**Team**: Ferel, Safa — ITS Informatics | KCVanguard ML Workshop.",
                ])
            )

    return demo


DEMO = build_demo()
DEMO.queue()

if __name__ == "__main__":
    DEMO.launch()