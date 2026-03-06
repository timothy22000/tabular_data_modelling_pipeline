#!/usr/bin/env python3
"""Generate professional architecture diagrams using ReportLab.

Creates high-quality vector-style PNG diagrams for:
  1. CANN  — Combined Actuarial Neural Network
  2. FT-Transformer — Feature Tokenizer Transformer
  3. TabM  — Tabular MLP Ensemble

Output:
    data_to_be_cleaned/net/dl_results/presentation/generated_figures/
        cann_architecture_v2.png
        ft_transformer_architecture_v2.png
        tabm_architecture_v2.png

Run with:
    conda activate video-to-text
    python create_architecture_diagrams.py
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.graphics import renderPM
from reportlab.graphics.shapes import (
    Drawing,
    Group,
    Line,
    Polygon,
    Rect,
    String,
)
from reportlab.lib.colors import Color, HexColor, white

# ---------------------------------------------------------------------------
# Palette (Navy / Gold — matches StyleKit)
# ---------------------------------------------------------------------------

NAVY = HexColor("#1B2A4A")
BLUE = HexColor("#2E6B9E")
GOLD = HexColor("#C8963E")
GREEN = HexColor("#1D9A6C")
RED = HexColor("#DC2626")
PURPLE = HexColor("#7C3AED")
WHITE = HexColor("#FFFFFF")
LIGHT_GRAY = HexColor("#F5F5F5")
DARK_GRAY = HexColor("#333333")
MID_GRAY = HexColor("#888888")
ROW_ALT = HexColor("#EBF0F7")

CANVAS_BG = WHITE
DPI = 200

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data_to_be_cleaned"
    / "net"
    / "dl_results"
    / "presentation"
    / "generated_figures"
)


# ---------------------------------------------------------------------------
# Drawing Helpers
# ---------------------------------------------------------------------------


def rounded_rect(
    d: Drawing | Group,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: Color,
    stroke: Color = DARK_GRAY,
    radius: float = 10,
    stroke_width: float = 1.5,
) -> None:
    """Add a rounded rectangle to a Drawing or Group."""
    r = Rect(x, y, w, h, rx=radius, ry=radius)
    r.fillColor = fill
    r.strokeColor = stroke
    r.strokeWidth = stroke_width
    d.add(r)


def label(
    d: Drawing | Group,
    x: float,
    y: float,
    text: str,
    size: float = 12,
    color: Color = WHITE,
    bold: bool = True,
    anchor: str = "middle",
) -> None:
    """Add a text label."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    s = String(x, y, text, fontName=font, fontSize=size, fillColor=color)
    s.textAnchor = anchor
    d.add(s)


def arrow(
    d: Drawing | Group,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Color = DARK_GRAY,
    lw: float = 1.8,
    head_size: float = 8,
) -> None:
    """Draw a line with an arrowhead at (x2, y2)."""
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=lw)
    d.add(line)

    # Arrowhead triangle
    angle = math.atan2(y2 - y1, x2 - x1)
    s = head_size
    # Three points of the arrowhead
    tip_x, tip_y = x2, y2
    left_x = tip_x - s * math.cos(angle - math.pi / 6)
    left_y = tip_y - s * math.sin(angle - math.pi / 6)
    right_x = tip_x - s * math.cos(angle + math.pi / 6)
    right_y = tip_y - s * math.sin(angle + math.pi / 6)

    head = Polygon(
        [tip_x, tip_y, left_x, left_y, right_x, right_y],
        fillColor=color,
        strokeColor=color,
        strokeWidth=0.5,
    )
    d.add(head)


def box(
    d: Drawing | Group,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text_main: str,
    fill: Color,
    text_color: Color = WHITE,
    font_size: float = 12,
    sublabel_text: str = "",
    sublabel_size: float = 10,
    stroke: Color = DARK_GRAY,
    radius: float = 10,
) -> None:
    """Draw a rounded box with centred label + optional sublabel."""
    bx = cx - w / 2
    by = cy - h / 2
    rounded_rect(d, bx, by, w, h, fill=fill, stroke=stroke, radius=radius)

    # Main label — shift up if sublabel present
    ty = cy + (6 if sublabel_text else -font_size * 0.35)
    label(d, cx, ty, text_main, size=font_size, color=text_color, bold=True)

    if sublabel_text:
        label(
            d, cx, cy - sublabel_size - 2,
            sublabel_text, size=sublabel_size,
            color=text_color, bold=False,
        )


def annotation_box(
    d: Drawing | Group,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    font_size: float = 11,
) -> None:
    """Draw a light annotation box with navy text and gold border."""
    bx = cx - w / 2
    by = cy - h / 2
    rounded_rect(d, bx, by, w, h, fill=ROW_ALT, stroke=GOLD, radius=8, stroke_width=2)
    label(d, cx, cy - font_size * 0.35, text, size=font_size, color=NAVY, bold=True)


# ---------------------------------------------------------------------------
# Diagram 1: CANN
# ---------------------------------------------------------------------------


def draw_cann(output_path: Path) -> None:
    """CANN architecture: GLM + NN residual → multiply."""
    W, H = 1800, 780
    d = Drawing(W, H)
    # White background
    d.add(Rect(0, 0, W, H, fillColor=CANVAS_BG, strokeColor=CANVAS_BG))

    # Title
    label(d, W / 2, H - 40, "CANN Architecture: GLM Base + Neural Network Residual",
          size=18, color=DARK_GRAY)

    # --- Input Features ---
    box(d, 200, 530, 260, 120, "Input Features", BLUE,
        sublabel_text="age, car, credit,\nmileage, NCD...")

    # --- GLM (bottom path) ---
    box(d, 200, 250, 260, 110, "GLM", GOLD, text_color=DARK_GRAY,
        sublabel_text="Traditional Actuarial Model", sublabel_size=9)

    # Arrow: Input → GLM
    arrow(d, 200, 530 - 60, 200, 250 + 55 + 8)

    # --- NN MLP (top path) ---
    box(d, 660, 530, 300, 120, "Neural Network", BLUE,
        sublabel_text="MLP: 128 → 64 → 1", sublabel_size=10)
    label(d, 660, 530 - 40, "Learns residual correction", size=9, color=WHITE, bold=False)

    # Arrow: Input → NN
    arrow(d, 200 + 130, 530, 660 - 150 - 8, 530)

    # --- Clamp ---
    box(d, 920, 530, 140, 80, "Clamp", NAVY,
        sublabel_text="[-2, +2]", sublabel_size=10)

    # Arrow: NN → Clamp
    arrow(d, 660 + 150, 530, 920 - 70 - 8, 530)

    # --- exp(r) ---
    box(d, 1080, 530, 120, 80, "exp(r)", NAVY, font_size=13)

    # Arrow: Clamp → exp
    arrow(d, 920 + 70, 530, 1080 - 60 - 8, 530)

    # --- Multiply ---
    box(d, 1300, 390, 160, 100, "Multiply", NAVY,
        sublabel_text="GLM × exp(r)", sublabel_size=10)

    # Arrow: exp → Multiply (down-right)
    arrow(d, 1080, 530 - 40, 1300 - 10, 390 + 50 + 8)

    # Arrow: GLM → Multiply (long horizontal)
    arrow(d, 200 + 130, 250, 1300 - 80 - 8, 250)
    # Then up to Multiply
    arrow(d, 1300, 250, 1300, 390 - 50 - 8)

    # --- Premium output ---
    box(d, 1570, 390, 200, 100, "£ Premium", GREEN,
        sublabel_text="Final Prediction", sublabel_size=10, font_size=15)

    # Arrow: Multiply → Premium
    arrow(d, 1300 + 80, 390, 1570 - 100 - 8, 390)

    # --- Arrow annotation labels ---
    label(d, 430, 555, "features", size=9, color=MID_GRAY, bold=False)
    label(d, 700, 235, "GLM prediction (baseline)", size=9, color=MID_GRAY, bold=False)
    label(d, 1180, 460, "correction", size=9, color=MID_GRAY, bold=False)
    label(d, 1180, 445, "factor", size=9, color=MID_GRAY, bold=False)

    # --- Formula annotation ---
    annotation_box(d, W / 2, 80, 700, 50,
                   "Final = GLM_prediction  ×  exp( clamp( NN_output,  -2,  +2 ) )",
                   font_size=13)

    _save(d, output_path)


# ---------------------------------------------------------------------------
# Diagram 2: FT-Transformer
# ---------------------------------------------------------------------------


def draw_ft_transformer(output_path: Path) -> None:
    """FT-Transformer: tokenize → self-attention → predict."""
    W, H = 1800, 850
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=CANVAS_BG, strokeColor=CANVAS_BG))

    # Title
    label(d, W / 2, H - 40,
          "FT-Transformer Architecture: Tokenize → Attend → Predict",
          size=18, color=DARK_GRAY)

    # --- Stage 1: Raw feature boxes ---
    features = ["Age\n35", "Mileage\n12K", "Credit\n720",
                "Fuel\nPetrol", "NCD\n5", "[CLS]\n?"]
    feat_colors = [BLUE] * 5 + [GREEN]
    n = len(features)
    spacing = 220
    start_x = (W - (n - 1) * spacing) / 2

    feature_y = H - 130
    for i, (feat, col) in enumerate(zip(features, feat_colors)):
        fx = start_x + i * spacing
        # Split multiline: show name above, value below
        parts = feat.split("\n")
        box(d, fx, feature_y, 170, 70, parts[0], col,
            sublabel_text=parts[1] if len(parts) > 1 else "", sublabel_size=11)

    # Stage label
    label(d, W - 60, feature_y, "STAGE 1", size=10, color=GOLD, bold=True)
    label(d, W - 60, feature_y - 16, "Tokenize", size=10, color=GOLD, bold=False)

    # Left label
    label(d, 45, feature_y, "Raw", size=10, color=DARK_GRAY, bold=True, anchor="end")
    label(d, 45, feature_y - 14, "Features", size=10, color=DARK_GRAY, bold=True, anchor="end")

    # --- Arrows down to tokens ---
    token_y = feature_y - 120
    for i in range(n):
        fx = start_x + i * spacing
        arrow(d, fx, feature_y - 35 - 8, fx, token_y + 30 + 8)

    # --- Token boxes ---
    for i, col in enumerate(feat_colors):
        fx = start_x + i * spacing
        box(d, fx, token_y, 170, 50, "64-d token", col, font_size=10)

    label(d, 45, token_y, "Token", size=10, color=DARK_GRAY, bold=True, anchor="end")
    label(d, 45, token_y - 14, "Vectors", size=10, color=DARK_GRAY, bold=True, anchor="end")

    # --- Arrows into transformer ---
    transformer_y = token_y - 120
    for i in range(n):
        fx = start_x + i * spacing
        arrow(d, fx, token_y - 25 - 8, fx, transformer_y + 40 + 8)

    # --- Transformer encoder bar ---
    bar_w = 1500
    bar_h = 80
    bar_x = (W - bar_w) / 2
    bar_y = transformer_y - bar_h / 2
    rounded_rect(d, bar_x, bar_y, bar_w, bar_h, fill=NAVY, stroke=GOLD,
                 radius=12, stroke_width=2.5)
    label(d, W / 2, transformer_y + 5,
          "Transformer Encoder  (3 layers × 4 attention heads)", size=13, color=WHITE)
    label(d, W / 2, transformer_y - 16,
          "Every token attends to every other token", size=11, color=WHITE, bold=False)

    # Stage label
    label(d, W - 60, transformer_y + 8, "STAGE 2", size=10, color=GOLD, bold=True)
    label(d, W - 60, transformer_y - 8, "Self-Attention", size=9, color=GOLD, bold=False)

    # --- Arrow from transformer to CLS output ---
    predict_y = transformer_y - 140
    arrow(d, W / 2, transformer_y - bar_h / 2 - 8, W / 2, predict_y + 30 + 8)

    # --- Stage 3: CLS → Head → Softplus → Premium ---
    step_xs = [500, 750, 1000, 1300]
    step_labels = ["[CLS] output", "MLP Head", "Softplus", "£ Premium"]
    step_colors = [GREEN, NAVY, NAVY, GREEN]
    step_widths = [180, 150, 140, 180]
    step_sizes = [12, 12, 11, 14]

    for sx, sl, sc, sw, ss in zip(step_xs, step_labels, step_colors, step_widths, step_sizes):
        box(d, sx, predict_y, sw, 60, sl, sc, font_size=ss)

    # Arrows between steps
    for i in range(len(step_xs) - 1):
        x1 = step_xs[i] + step_widths[i] / 2
        x2 = step_xs[i + 1] - step_widths[i + 1] / 2
        arrow(d, x1, predict_y, x2 - 8, predict_y)

    # Stage label
    label(d, W - 60, predict_y + 8, "STAGE 3", size=10, color=GOLD, bold=True)
    label(d, W - 60, predict_y - 8, "Predict", size=10, color=GOLD, bold=False)

    # --- Annotation ---
    annotation_box(d, W / 2, 50, 820, 46,
                   "Each feature is treated like a word — self-attention discovers "
                   "feature interactions automatically",
                   font_size=11)

    _save(d, output_path)


# ---------------------------------------------------------------------------
# Diagram 3: TabM
# ---------------------------------------------------------------------------


def draw_tabm(output_path: Path) -> None:
    """TabM: ensemble of K independent MLPs with soft averaging."""
    W, H = 1800, 900
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=CANVAS_BG, strokeColor=CANVAS_BG))

    # Title
    label(d, W / 2, H - 40, "TabM Architecture: Ensemble of K Independent MLPs",
          size=18, color=DARK_GRAY)

    # --- Shared Input ---
    input_y = H - 140
    box(d, W / 2, input_y, 620, 80, "Shared Input Features", BLUE,
        sublabel_text="+ Categorical Embeddings", sublabel_size=11, font_size=13)

    # --- MLP members ---
    mlp_labels = ["MLP 1", "MLP 2", "MLP 3", "...", "MLP 16"]
    mlp_colors = [BLUE, GREEN, GOLD, None, PURPLE]
    mlp_preds = ["£923", "£987", "£1,041", "", "£956"]
    n_mlps = len(mlp_labels)
    mlp_spacing = 300
    mlp_start_x = (W - (n_mlps - 1) * mlp_spacing) / 2
    mlp_y = input_y - 220

    # Fan-out arrows
    for i in range(n_mlps):
        mx = mlp_start_x + i * mlp_spacing
        if mlp_labels[i] == "...":
            continue
        arrow(d, W / 2, input_y - 40 - 8, mx, mlp_y + 55 + 8)

    # MLP boxes
    for i, (mlabel, col, pred) in enumerate(zip(mlp_labels, mlp_colors, mlp_preds)):
        mx = mlp_start_x + i * mlp_spacing
        if mlabel == "...":
            label(d, mx, mlp_y + 5, "· · ·", size=24, color=MID_GRAY)
            continue
        box(d, mx, mlp_y, 220, 100, mlabel, col,
            sublabel_text="128 → 64 → 1", sublabel_size=10, font_size=13)
        # Prediction label below
        label(d, mx, mlp_y - 65, pred, size=12, color=col, bold=True)

    # --- Fan-in arrows ---
    avg_y = mlp_y - 190
    for i in range(n_mlps):
        mx = mlp_start_x + i * mlp_spacing
        if mlp_labels[i] == "...":
            continue
        arrow(d, mx, mlp_y - 50 - 20 - 8, W / 2, avg_y + 40 + 8)

    # --- Soft Average ---
    box(d, W / 2, avg_y, 500, 80, "Learned Soft Average", NAVY,
        sublabel_text="softmax weights", sublabel_size=11, font_size=13)

    # --- Arrow to final ---
    final_y = avg_y - 130
    arrow(d, W / 2, avg_y - 40 - 8, W / 2, final_y + 35 + 8)

    # --- Final prediction ---
    box(d, W / 2, final_y, 240, 65, "£978 Final", GREEN, font_size=15)

    # --- Annotation ---
    annotation_box(d, W / 2, 50, 900, 46,
                   "Each MLP learns independently — diversity from random "
                   "initialisation.  Learned weights determine each member's influence.",
                   font_size=11)

    _save(d, output_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(d: Drawing, path: Path) -> None:
    """Render a Drawing to PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    renderPM.drawToFile(d, str(path), fmt="PNG", dpi=DPI)
    size_kb = path.stat().st_size / 1024
    print(f"  [OK] {path.name}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    separator = "=" * 56
    print(separator)
    print("  Architecture Diagram Generator (ReportLab)")
    print(separator)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    draw_cann(OUTPUT_DIR / "cann_architecture_v2.png")
    draw_ft_transformer(OUTPUT_DIR / "ft_transformer_architecture_v2.png")
    draw_tabm(OUTPUT_DIR / "tabm_architecture_v2.png")

    print(separator)
    print(f"  Output: {OUTPUT_DIR}")
    print(separator)


if __name__ == "__main__":
    main()
