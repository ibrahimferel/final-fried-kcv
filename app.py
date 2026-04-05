"""
Module: app
SRS Reference: FR-DEP-001–010
SDLC Phase: 4 - Implementation (Sprint E)
Sprint: E
Pipeline Stage: Deployment
Purpose: Gradio 4.x UI wiring for the full DSDBA pipeline with non-blocking NLP explanation.
Dependencies: gradio, asyncio, PyYAML, torch, onnxruntime, matplotlib
Interface Contract:
  Input: WAV/FLAC filepath (<= 20 MB) from Gradio upload
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
from typing import Any, Generator, Iterable

import gradio as gr
import numpy as np
import torch
import yaml

from src.audio.dsp import preprocess_audio
from src.cv.gradcam import run_gradcam
from src.cv.infer import export_to_onnx, load_onnx_session, run_onnx_inference
from src.cv.model import DSDBAModel
from src.nlp.explain import generate_explanation
from src.utils.errors import DSDBAError
from src.utils.logger import log_error, log_info, log_warning


def load_config(path: str | Path) -> dict[str, Any]:
  """
  Load config.yaml from disk.

  Args:
    path: Path to config YAML file.

  Returns:
    Parsed configuration mapping.

  Raises:
    FileNotFoundError: If config path does not exist.
    ValueError: If YAML cannot be parsed.
  """
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
  """Return project root directory (repo root)."""
  return Path(__file__).resolve().parent


def _models_dir() -> Path:
  """Return models directory under project root."""
  return _project_root() / "models" / "checkpoints"


def _ensure_onnx_session(cfg: dict[str, Any], model: DSDBAModel) -> Any:
  """
  Create an ONNX Runtime session once at startup.

  Args:
    cfg: Full configuration mapping.
    model: DSDBA PyTorch model used for ONNX export if needed.

  Returns:
    onnxruntime.InferenceSession
  """
  _models_dir().mkdir(parents=True, exist_ok=True)
  onnx_path = _models_dir() / "dsdba_efficientnet_b4.onnx"
  if not onnx_path.exists():
    log_warning(stage="deployment", message="onnx_missing_exporting", data={"path": str(onnx_path)})
    onnx_path = export_to_onnx(model=model, cfg=cfg)
  return load_onnx_session(onnx_path=onnx_path, cfg=cfg)


def _maybe_load_weights(model: DSDBAModel, cfg: dict[str, Any]) -> None:
  """
  Load best checkpoint weights if present; otherwise keep random weights.

  Args:
    model: DSDBAModel instance.
    cfg: Full configuration mapping.

  Returns:
    None.
  """
  ckpt_name = str(cfg.get("training", {}).get("best_checkpoint_filename", "best_model.pth"))
  ckpt_path = _models_dir() / ckpt_name
  if not ckpt_path.exists():
    log_warning(stage="deployment", message="checkpoint_missing_random_weights", data={"path": str(ckpt_path)})
    return

  try:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)
    log_info(stage="deployment", message="checkpoint_loaded", data={"path": str(ckpt_path)})
  except Exception as exc:
    log_warning(stage="deployment", message="checkpoint_load_failed_random_weights", data={"path": str(ckpt_path), "reason": str(exc)})


def _validate_file_size(audio_path: Path, cfg: dict[str, Any]) -> None:
  """
  Validate upload size before any processing (FR-DEP-002).

  Args:
    audio_path: Path to the uploaded audio file.
    cfg: Full config mapping (uses deployment.max_upload_mb).

  Returns:
    None.

  Raises:
    ValueError: If file exceeds configured size.
  """
  max_mb = float(cfg["deployment"]["max_upload_mb"])
  size_bytes = int(audio_path.stat().st_size)
  if size_bytes > int(max_mb * 1024 * 1024):
    raise ValueError("FILE_TOO_LARGE")


def _band_df(band_pct: dict[str, float]) -> dict[str, list[Any]]:
  """
  Convert band_pct dict into a dataframe-like dict for gr.BarPlot without pandas dependency.

  Args:
    band_pct: Mapping of band name -> percent.

  Returns:
    Dict with columns: band, percent.
  """
  order = ["low", "low_mid", "high_mid", "high"]
  bands = [b for b in order if b in band_pct]
  perc = [float(band_pct[b]) for b in bands]
  return {"band": bands, "percent": perc}

def _band_plot(band_pct: dict[str, float]):
    import matplotlib.pyplot as plt
    order = ["low", "low_mid", "high_mid", "high"]
    bands = [b for b in order if b in band_pct]
    perc = [float(band_pct[b]) for b in bands]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(bands, perc, color="#6366f1")
    ax.set_ylabel("Attribution (%)")
    ax.set_title("Frequency Band Attribution")
    plt.tight_layout()
    return fig


def _confidence_percent(conf: float) -> float:
  """Convert confidence ratio (0..1) into percentage."""
  return float(conf) * 100.0


def _verdict_html(label: str, confidence: float) -> str:
  """Return a simple HTML bar showing confidence."""
  pct = max(0.0, min(100.0, _confidence_percent(confidence)))
  color = "#ef4444" if str(label).lower() == "spoof" else "#22c55e"
  return (
    "<div style='width: 100%; background: #e5e7eb; border-radius: 8px; overflow: hidden;'>"
    f"<div style='width: {pct:.2f}%; background: {color}; padding: 6px 0; color: white; text-align: center;'>"
    f"{pct:.2f}%"
    "</div></div>"
  )


def _spectrogram_image_from_tensor(tensor: torch.Tensor) -> Path:
  """
  Create a quick spectrogram image from the first channel of the input tensor.

  Args:
    tensor: torch.Tensor [3,224,224] float32.

  Returns:
    Path to a temporary PNG file.
  """
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
  """
  Ensure demo audio samples exist under data/samples (FR-DEP-008).

  Notes:
    These are synthetic tones/noise for UI smoke tests, not real dataset clips.

  Args:
    cfg: Full configuration mapping.

  Returns:
    List of 4 sample file paths.
  """
  import soundfile as sf

  root = _project_root() / "data" / "samples"
  root.mkdir(parents=True, exist_ok=True)
  sr = int(cfg["audio"]["sample_rate"])
  n = int(cfg["audio"]["n_samples"])
  t = np.linspace(0.0, float(cfg["audio"]["duration_sec"]), num=n, endpoint=False, dtype=np.float32)

  samples: list[tuple[str, np.ndarray]] = []
  # "Bonafide": clean tones
  samples.append(("bonafide_01.wav", 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)))
  samples.append(("bonafide_02.wav", 0.1 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)))
  # "Spoof": noisy / high-frequency emphasis (synthetic)
  rng = np.random.default_rng(0)
  noise = (0.03 * rng.standard_normal(size=n)).astype(np.float32)
  hf = (0.06 * np.sin(2.0 * np.pi * 3200.0 * t)).astype(np.float32)
  samples.append(("spoof_01.wav", np.clip(noise + hf, -1.0, 1.0)))
  samples.append(("spoof_02.wav", np.clip(noise + 0.06 * np.sin(2.0 * np.pi * 5200.0 * t).astype(np.float32), -1.0, 1.0)))

  paths: list[Path] = []
  for name, wav in samples:
    p = root / name
    if not p.exists():
      sf.write(str(p), wav, sr, subtype="PCM_16")
    paths.append(p)
  return paths


def run_pipeline(audio_file: str | Path, cfg: dict[str, Any], onnx_session: Any, model: DSDBAModel):
  """
  MAIN PIPELINE FUNCTION (FR-DEP-002..010).

  Stages:
    1) Preprocess audio -> tensor (FR-AUD-001..008)
    2) ONNX inference -> (label, confidence) (FR-CV-004, FR-DEP-010)
    3) Grad-CAM -> heatmap_path, band_pct (FR-CV-010..016)
    4) NLP explanation started as asyncio.Task (FR-NLP-006)

  Args:
    audio_file: File path from Gradio upload.
    cfg: Full configuration mapping.
    onnx_session: ONNX Runtime session created once at startup.
    model: DSDBAModel used for Grad-CAM.

  Returns:
    (label, confidence, spectrogram_path, heatmap_path, band_pct, explanation_task)
  """
  audio_path = Path(audio_file)
  _validate_file_size(audio_path=audio_path, cfg=cfg)

  tensor = preprocess_audio(file_path=audio_path, cfg=cfg)
  label, confidence = run_onnx_inference(session=onnx_session, tensor=tensor, cfg=cfg)
  heatmap_path, band_pct = run_gradcam(tensor=tensor, model=model, cfg=cfg)

  # Start NLP without blocking the CV result display (FR-NLP-006).
  explanation_task = asyncio.create_task(generate_explanation(label=label, confidence=confidence, band_pct=band_pct, cfg=cfg))
  spec_path = _spectrogram_image_from_tensor(tensor)
  return label, confidence, spec_path, heatmap_path, band_pct, explanation_task


# ── Startup: load config + model + ONNX session ONCE ──────────────────────────
CFG: dict[str, Any] = load_config("config.yaml")
MODEL: DSDBAModel = DSDBAModel(cfg=CFG, pretrained=False)
_maybe_load_weights(model=MODEL, cfg=CFG)
MODEL.eval()
ONNX_SESSION = _ensure_onnx_session(cfg=CFG, model=MODEL)


def _ui_error_outputs() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
  """Return empty placeholders for UI outputs."""
  return (None, None, "", None, None, None, None, "")


async def ui_run(audio_path: str | None) -> Generator[tuple[Any, Any, Any, Any, Any, Any, Any, Any], None, None]:
  """
  Gradio handler that streams outputs so CV results appear before NLP finishes (FR-NLP-006).

  Outputs:
    - verdict label (str)
    - confidence percent (float)
    - confidence bar HTML (str)
    - waveform display: use the original path (str)
    - spectrogram image path (Path)
    - gradcam overlay image path (Path)
    - band barplot data (dict)
    - explanation textbox (str)
  """
  if not audio_path:
    gr.Warning("Please upload a WAV/FLAC file.")
    yield _ui_error_outputs()
    return

  start = time.perf_counter()
  try:
    label, confidence, spec_path, heatmap_path, band_pct, explanation_task = run_pipeline(
      audio_file=audio_path,
      cfg=CFG,
      onnx_session=ONNX_SESSION,
      model=MODEL,
    )

    # First yield: show CV+XAI outputs immediately, keep explanation "loading".
    yield (
      label,
      _confidence_percent(confidence),
      _verdict_html(label=label, confidence=confidence),
      str(audio_path),
      spec_path,
      heatmap_path,
      _band_plot(band_pct),
      "Generating explanation…",
    )

    explanation_text, api_was_used = await explanation_task
    if not api_was_used:
      gr.Warning(str(CFG["nlp"].get("warning_badge_text", "AI explanation unavailable")))

    yield (
      label,
      _confidence_percent(confidence),
      _verdict_html(label=label, confidence=confidence),
      str(audio_path),
      spec_path,
      heatmap_path,
      _band_plot(band_pct),
      explanation_text,
    )
  except ValueError as exc:
    if str(exc) == "FILE_TOO_LARGE":
      gr.Warning("File too large (> 20 MB). Please upload a smaller WAV/FLAC.")
      yield _ui_error_outputs()
      return
    gr.Error("Failed to process input.")
    log_error(stage="deployment", message="ui_value_error", data={"reason": str(exc)})
    yield _ui_error_outputs()
  except DSDBAError as exc:
    if exc.code == "AUD-001":
      gr.Warning("Audio too short (< 0.5 s).")
    elif exc.code == "AUD-002":
      gr.Warning("Unsupported format. Please upload WAV or FLAC.")
    else:
      gr.Warning("Audio processing failed.")
    yield _ui_error_outputs()
  except Exception as exc:
    gr.Error("Unexpected error. Please try again.")
    log_error(stage="deployment", message="ui_exception", data={"reason": str(exc)})
    yield _ui_error_outputs()
  finally:
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log_info(stage="deployment", message="ui_run_complete", data={"latency_ms": round(elapsed_ms, 3)})


def build_demo() -> gr.Blocks:
  """
  Build the Gradio Blocks UI (FR-DEP-001..009).

  Returns:
    Gradio Blocks demo instance.
  """
  demo_samples = ensure_demo_samples(CFG)

  with gr.Blocks(title="DSDBA — Deepfake Speech Detection") as demo:
    gr.Markdown("## DSDBA — Deepfake Speech Detection & Explainability (Gradio 4.x)")
    gr.Markdown("Upload a WAV/FLAC file (≤ 20 MB). CV results appear first; explanation loads asynchronously.")

    with gr.Row():
      with gr.Column(scale=1):
        audio_in = gr.Audio(label="Upload audio", type="filepath")
        run_btn = gr.Button("Run")

      with gr.Column(scale=2):
        with gr.Row():
          verdict = gr.Label(label="Verdict")
          confidence_pct = gr.Number(label="Confidence (%)", precision=2)
        conf_bar = gr.HTML(label="Confidence bar")

        waveform = gr.Audio(label="Waveform", type="filepath")
        spec_img = gr.Image(label="Spectrogram (proxy)", type="filepath")

        gradcam_img = gr.Image(label="Grad-CAM overlay", type="filepath")
        band_plot = gr.Plot(label="Band attribution (%)")

        gr.Markdown("**AI-generated explanation (English)**")
        explanation = gr.Textbox(label="Explanation", lines=6)

    run_btn.click(
      fn=ui_run,
      inputs=[audio_in],
      outputs=[verdict, confidence_pct, conf_bar, waveform, spec_img, gradcam_img, band_plot, explanation],
    )

    with gr.Accordion("About", open=False):
      gr.Markdown(
        "\n".join(
          [
            "### About",
            "- **Pipeline**: Audio DSP → EfficientNet-B4 (ONNX) → Grad-CAM → LLM explanation (with rule-based fallback).",
            "- **Dataset citation**: Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.",
            "- **Team**: Ferel, Safa — ITS Informatics | KCVanguard ML Workshop.",
          ]
        )
      )

  return demo


DEMO = build_demo()
DEMO.launch(server_name="0.0.0.0", server_port=7860, show_error=True)