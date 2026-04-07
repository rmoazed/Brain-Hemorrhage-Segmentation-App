import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st


# ============================================================
# Brain Hemorrhage Segmentation Explorer
# ------------------------------------------------------------
# This app is designed around PRECOMPUTED examples, so it does
# not run model inference live. Instead, it loads saved images,
# true masks, and predicted probability maps / masks.
#
# Expected folder structure:
#
# project_root/
#   brain_hemorrhage_segmentation_explorer.py
#   app_data/
#     summary.json                      # optional overall metrics
#     results_table.csv                # optional model results table
#     cases/
#       case_001/
#         input.png
#         true_mask.png
#         unet_prob.npy                # preferred (probability map)
#         uresnet_prob.npy             # preferred (probability map)
#         # OR fallback binary masks if npy files are unavailable:
#         unet_mask.png
#         uresnet_mask.png
#         metadata.json                # optional per-case notes
#       case_002/
#         ...
#
# Notes:
# - If a *_prob.npy file exists, the threshold slider will be used.
# - If only *_mask.png exists, that saved mask will be displayed.
# - You can rename files if needed, but then also update the paths below.
# ============================================================

APP_DATA_DIR = Path("app_data")
CASES_DIR = APP_DATA_DIR / "cases"
SUMMARY_PATH = APP_DATA_DIR / "summary.json"
RESULTS_TABLE_PATH = APP_DATA_DIR / "results_table.csv"

MODEL_FILE_MAP = {
    "U-Net": {
        "prob": "unet_prob.npy",
        "mask": "unet_mask.png",
    },
    "UResNet": {
        "prob": "uresnet_prob.npy",
        "mask": "uresnet_mask.png",
    },
}


def load_image(path: Path) -> Image.Image:
    return Image.open(path)


def load_mask_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_probability_map(case_dir: Path, model_name: str) -> Optional[np.ndarray]:
    prob_path = case_dir / MODEL_FILE_MAP[model_name]["prob"]
    if prob_path.exists():
        return np.load(prob_path)
    return None


def load_saved_mask(case_dir: Path, model_name: str) -> Optional[np.ndarray]:
    mask_path = case_dir / MODEL_FILE_MAP[model_name]["mask"]
    if mask_path.exists():
        return load_mask_png(mask_path)
    return None


def get_case_dirs() -> list[Path]:
    if not CASES_DIR.exists():
        return []
    return sorted([
        p for p in CASES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith('.')])


def read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_prob_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def to_binary_mask(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map > threshold).astype(np.uint8)


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8)

    overlay = image_arr.copy()
    # Red overlay for predicted hemorrhage region
    overlay[mask == 1] = [255, 0, 0]

    blended = (image_arr * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def main() -> None:
    st.set_page_config(page_title="Brain Hemorrhage Segmentation Explorer", layout="wide")

    st.title("Brain Hemorrhage Segmentation Explorer")
    st.markdown(
        "Explore precomputed CT examples, compare true masks with model outputs, "
        "and interactively adjust segmentation thresholds when probability maps are available."
    )

    summary = read_json_if_exists(SUMMARY_PATH)
    case_dirs = get_case_dirs()

    if not case_dirs:
        st.error(
            "No case folders were found. Create an `app_data/cases/` folder and add at least one case directory."
        )
        st.stop()

    with st.sidebar:
        st.header("Controls")

        case_names = [p.name for p in case_dirs]
        selected_case_name = st.selectbox("Select case", case_names)
        selected_case_dir = next(p for p in case_dirs if p.name == selected_case_name)

        model_name = st.selectbox("Select model", list(MODEL_FILE_MAP.keys()))
        threshold = st.slider("Threshold", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
        show_overlay = st.checkbox("Show overlay on input image", value=True)
        show_raw_probability = st.checkbox("Show raw probability map (if available)", value=True)

        st.markdown("---")
        st.caption("Tip: If you only saved binary masks instead of probability maps, the threshold slider will not affect that model output.")

    input_path = selected_case_dir / "input.png"
    true_mask_path = selected_case_dir / "true_mask.png"
    metadata_path = selected_case_dir / "metadata.json"

    if not input_path.exists() or not true_mask_path.exists():
        st.error(f"Case `{selected_case_name}` is missing `input.png` or `true_mask.png`.")
        st.stop()

    input_image = load_image(input_path)
    true_mask = load_mask_png(true_mask_path)

    prob_map = load_probability_map(selected_case_dir, model_name)
    saved_mask = load_saved_mask(selected_case_dir, model_name)

    if prob_map is None and saved_mask is None:
        st.error(
            f"No prediction found for {model_name} in case `{selected_case_name}`. "
            f"Expected either `{MODEL_FILE_MAP[model_name]['prob']}` or `{MODEL_FILE_MAP[model_name]['mask']}`."
        )
        st.stop()

    if prob_map is not None:
        prob_map = normalize_prob_map(prob_map)
        pred_mask = to_binary_mask(prob_map, threshold)
        prediction_mode = "Probability map + threshold"
    else:
        pred_mask = (saved_mask > 0).astype(np.uint8)
        prediction_mode = "Saved binary mask"

    metadata = read_json_if_exists(metadata_path)

    if summary:
        st.subheader("Project Summary")
        metric_cols = st.columns(4)
        for idx, (key, value) in enumerate(summary.items()):
            metric_cols[idx % 4].metric(label=str(key), value=str(value))

    st.subheader("Selected Example")
    if metadata:
        with st.expander("Case metadata", expanded=False):
            st.json(metadata)

    st.caption(f"Case: {selected_case_name} | Model: {model_name} | Prediction mode: {prediction_mode}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Input CT image**")
        st.image(input_image, use_column_width=True)

        if show_overlay:
            st.markdown("**Predicted mask overlay**")
            overlay_img = overlay_mask_on_image(input_image, pred_mask)
            st.image(overlay_img, use_column_width=True)

    with col2:
        st.markdown("**True mask**")
        st.image(true_mask, clamp=True, use_column_width=True)

        st.markdown("**Predicted binary mask**")
        st.image(pred_mask * 255, clamp=True, use_column_width=True)

    with col3:
        if prob_map is not None and show_raw_probability:
            st.markdown("**Raw probability map**")
            st.image(prob_map, clamp=True, use_column_width=True)

        st.markdown("**Threshold info**")
        if prob_map is not None:
            st.write(
                {
                    "threshold": threshold,
                    "pred_min": float(np.min(prob_map)),
                    "pred_mean": float(np.mean(prob_map)),
                    "pred_max": float(np.max(prob_map)),
                }
            )
        else:
            st.write("Using saved binary mask; no probability map was provided for this model/case.")

    st.markdown("---")
    st.subheader("How to use this demo")
    st.markdown(
        "- Switch between cases in the sidebar.\n"
        "- Compare U-Net and UResNet outputs.\n"
        "- Adjust the threshold when probability maps are available.\n"
        "- Use overlays to see where predicted hemorrhage regions fall on the scan."
    )


if __name__ == "__main__":
    main()

