from __future__ import annotations

import base64
import html
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

TEXT = "#1F2528"
TEXT_SECONDARY = "#5E6468"
TEAL = "#0F5B5B"
TEAL_ALT = "#1F6F6B"
COPPER = "#B76538"
GOLD = "#D6A63A"
BG = "#FAF8F3"
CARD_BG = "#FDFBF7"
BORDER = "#E4DDD2"
TABLE_ZEBRA = "#F4EFE7"
WHITE = "#FFFFFF"


def _ensure_rgb(image: Any) -> np.ndarray:
    if image is None:
        return np.full((220, 220, 3), 250, dtype=np.uint8)

    array = np.asarray(image)
    if array.size == 0:
        return np.full((220, 220, 3), 250, dtype=np.uint8)

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(array)) <= 1.5:
            array = array * 255.0

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 4:
        array = array[..., :3]

    return np.clip(array, 0, 255).astype(np.uint8)


def _wrap_text(text: Any, width: int = 92) -> str:
    value = str(text).strip()
    if not value:
        return ""
    return "\n".join(wrap(value, width=width))


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _clean_lines(lines: Iterable[Any]) -> list[str]:
    return [str(line).strip() for line in lines if str(line).strip()]


def _image_to_data_uri(image: Any) -> str | None:
    if image is None:
        return None
    rgb = _ensure_rgb(image)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_to_data_uri(fig: plt.Figure, fmt: str = "svg") -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/svg+xml" if fmt == "svg" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _style_chart(ax: plt.Axes, title: str, xlabel: str | None = None) -> None:
    ax.set_facecolor(WHITE)
    ax.set_title(title, fontsize=11.2, fontweight="bold", color=TEXT, loc="left", pad=10)
    ax.grid(axis="y", color=BORDER, linewidth=0.8, alpha=0.9)
    ax.tick_params(labelsize=8.6, colors=TEXT_SECONDARY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.8, color=TEXT_SECONDARY, labelpad=8)


def _safe_numeric_series(values: Sequence[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if series.isna().all():
        return np.arange(len(series), dtype=float)
    return series.ffill().fillna(0.0).to_numpy(dtype=float)


def _label_line_end(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, label: str, color: str) -> None:
    if len(x_values) == 0 or len(y_values) == 0:
        return
    offset = 0.015 * max(1.0, float(np.max(x_values) - np.min(x_values) if len(x_values) > 1 else 1.0))
    ax.text(
        float(x_values[-1]) + offset,
        float(y_values[-1]),
        label,
        fontsize=8.6,
        color=color,
        ha="left",
        va="center",
    )


def _build_metrics_curve_uri(curve_df: pd.DataFrame) -> str | None:
    if curve_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    fig.patch.set_facecolor(WHITE)
    x_values = _safe_numeric_series(curve_df.index.tolist())
    _style_chart(ax, "Deletion / Insertion Curves", xlabel=str(curve_df.index.name or "Fraction"))

    if "Deletion" in curve_df.columns:
        deletion = _safe_numeric_series(curve_df["Deletion"].tolist())
        ax.plot(x_values, deletion, color=COPPER, linewidth=2.0, label="Deletion")
        _label_line_end(ax, x_values, deletion, "Deletion", COPPER)
    if "Insertion" in curve_df.columns:
        insertion = _safe_numeric_series(curve_df["Insertion"].tolist())
        ax.plot(x_values, insertion, color=TEAL_ALT, linewidth=2.0, label="Insertion")
        _label_line_end(ax, x_values, insertion, "Insertion", TEAL_ALT)

    return _figure_to_data_uri(fig, fmt="svg")


def _build_counterfactual_curve_uri(progression_df: pd.DataFrame) -> str | None:
    required = {
        "Removed Area (%)",
        "Original Class Probability (%)",
        "Current Winner Probability (%)",
    }
    if progression_df.empty or not required.issubset(progression_df.columns):
        return None

    fig, ax = plt.subplots(figsize=(7.7, 2.7))
    fig.patch.set_facecolor(WHITE)
    x_values = _safe_numeric_series(progression_df["Removed Area (%)"].tolist())
    original = _safe_numeric_series(progression_df["Original Class Probability (%)"].tolist())
    winner = _safe_numeric_series(progression_df["Current Winner Probability (%)"].tolist())
    _style_chart(ax, "Confidence Trajectory", xlabel="Removed Area (%)")
    ax.plot(x_values, original, color=COPPER, linewidth=2.0)
    ax.plot(x_values, winner, color=TEAL_ALT, linewidth=2.0)
    _label_line_end(ax, x_values, original, "Original class", COPPER)
    _label_line_end(ax, x_values, winner, "Current winner", TEAL_ALT)
    return _figure_to_data_uri(fig, fmt="svg")


def _semantic_bar_items(semantic: Mapping[str, Any]) -> list[tuple[str, float]]:
    top_concepts = semantic.get("top_concepts")
    if isinstance(top_concepts, Sequence):
        items: list[tuple[str, float]] = []
        for concept in top_concepts[:6]:
            try:
                name, score = concept
                items.append((str(name), float(score)))
            except (TypeError, ValueError):
                continue
        if items:
            return items

    score_table = semantic.get("score_table")
    if isinstance(score_table, pd.DataFrame) and not score_table.empty:
        concept_col = None
        score_col = None
        for column in score_table.columns:
            lower = str(column).lower()
            if concept_col is None and ("concept" in lower or "έννο" in lower):
                concept_col = column
            if score_col is None and ("score" in lower or "%" in lower):
                score_col = column
        if concept_col is not None and score_col is not None:
            items: list[tuple[str, float]] = []
            for _, row in score_table.head(6).iterrows():
                try:
                    items.append((str(row[concept_col]), float(row[score_col])))
                except (TypeError, ValueError):
                    continue
            return items
    return []


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return '<div class="table-empty">Not available in this run.</div>'

    display_df = df.copy().head(max_rows).fillna("")
    headers = "".join(f"<th>{_escape(col)}</th>" for col in display_df.columns)
    body_rows: list[str] = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{_escape(value)}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _placeholder_data_uri(label: str, subtitle: str | None = None, width: int = 1200, height: int = 760) -> str:
    subtitle_value = subtitle.strip() if subtitle else ""
    subtitle_svg = (
        f'<text x="{width / 2:.1f}" y="{height / 2 + 42:.1f}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="28" fill="{TEXT_SECONDARY}">'
        f"{html.escape(subtitle_value)}</text>"
        if subtitle_value
        else ""
    )
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect width="100%" height="100%" fill="{CARD_BG}"/>
      <rect x="28" y="28" width="{width - 56}" height="{height - 56}" rx="26" fill="{WHITE}" stroke="{BORDER}" stroke-width="3"/>
      <line x1="28" y1="28" x2="{width - 28}" y2="28" stroke="{TEAL}" stroke-width="10" />
      <text x="{width / 2:.1f}" y="{height / 2 - 10:.1f}" text-anchor="middle"
            font-family="DejaVu Sans, Arial, sans-serif" font-size="38" font-weight="700" fill="{TEXT}">
        {html.escape(label)}
      </text>
      {subtitle_svg}
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def _slot_image_uri(image: Any, label: str, subtitle: str | None = None) -> str:
    return _image_to_data_uri(image) or _placeholder_data_uri(label, subtitle)


def _preferred_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    present = [column for column in columns if column in df.columns]
    return df[present].copy() if present else df.copy()


def _chips_html(labels: Sequence[str], availability: Mapping[str, bool]) -> str:
    chips: list[str] = []
    for label in labels:
        chip_class = "chip" if availability.get(label, False) else "chip muted"
        chips.append(f'<span class="{chip_class}">{_escape(label)}</span>')
    return "".join(chips)


def _bars_html(items: Sequence[tuple[str, float]]) -> str:
    cleaned: list[tuple[str, float]] = []
    for label, value in items[:6]:
        try:
            cleaned.append((str(label), float(value)))
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return '<div class="table-empty">Not available in this run.</div>'

    max_score = max(max(score for _, score in cleaned), 1.0)
    rows: list[str] = []
    for label, score in cleaned:
        width = max(8.0, min(100.0, (score / max_score) * 100.0))
        rows.append(
            f"""
            <div class="bar-row">
              <div>{_escape(label)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div>{score:.1f}%</div>
            </div>
            """
        )
    return "".join(rows)


def _metric_cards_html(metrics: Mapping[str, Any]) -> str:
    metric_hints = {
        "Deletion AUC": "Lower is better.",
        "Insertion AUC": "Higher is better.",
        "AOPC-like Delta": "Higher means stronger evidence impact.",
        "Sensitivity": "Lower suggests more stable attribution.",
        "Hoyer Sparsity": "Higher indicates concentrated attribution.",
        "Robustness": "Higher indicates more stable rankings.",
    }
    metric_items = [
        ("Deletion AUC", metrics.get("deletion_auc", "-") if metrics else "-"),
        ("Insertion AUC", metrics.get("insertion_auc", "-") if metrics else "-"),
        ("AOPC-like Delta", metrics.get("aopc_delta", "-") if metrics else "-"),
        ("Sensitivity", metrics.get("sensitivity", "-") if metrics else "-"),
        ("Hoyer Sparsity", metrics.get("hoyer_sparsity", "-") if metrics else "-"),
        ("Robustness", metrics.get("robustness", "-") if metrics else "-"),
    ]
    return "".join(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{_escape(label)}</div>
          <div class="kpi-value">{_escape(value)}</div>
          <div class="kpi-hint">{_escape(metric_hints[label])}</div>
        </div>
        """
        for label, value in metric_items
    )


def _build_html_report(payload: Mapping[str, Any]) -> str:
    report_title = str(payload.get("title", "XAI Analysis Report"))
    meta = dict(payload.get("meta", {}))
    overview = dict(payload.get("overview", {}))
    semantic = dict(payload.get("semantic", {}))
    metrics = dict(payload.get("metrics", {}))
    counterfactual = dict(payload.get("counterfactual", {}))
    shared_focus = dict(payload.get("shared_focus", {}))

    generated_at = datetime.now().strftime("%d %b %Y, %H:%M")

    top5_df = overview.get("top5_df") if isinstance(overview.get("top5_df"), pd.DataFrame) else pd.DataFrame()
    top5_df = _preferred_columns(top5_df, ["Rank", "Class Name", "Probability (%)"])
    semantic_df = semantic.get("score_table") if isinstance(semantic.get("score_table"), pd.DataFrame) else pd.DataFrame()
    metrics_df = metrics.get("details_df") if isinstance(metrics.get("details_df"), pd.DataFrame) else pd.DataFrame()
    pairwise_df = shared_focus.get("pairwise_df") if isinstance(shared_focus.get("pairwise_df"), pd.DataFrame) else pd.DataFrame()
    method_df = shared_focus.get("method_df") if isinstance(shared_focus.get("method_df"), pd.DataFrame) else pd.DataFrame()
    progression_df = counterfactual.get("progression_df") if isinstance(counterfactual.get("progression_df"), pd.DataFrame) else pd.DataFrame()
    curve_df = metrics.get("curve_df") if isinstance(metrics.get("curve_df"), pd.DataFrame) else pd.DataFrame()

    overview_original_uri = _slot_image_uri(overview.get("original_image"), "Original Image")
    overview_explained_uri = _slot_image_uri(overview.get("explained_image"), "Explained View")
    overview_heatmap_uri = _slot_image_uri(overview.get("heatmap_image"), "Raw Heatmap")
    semantic_focus_uri = _slot_image_uri(semantic.get("focus_image"), "Semantic Focus")
    cf_original_uri = _slot_image_uri(counterfactual.get("original_image"), "Original Image")
    cf_removed_uri = _slot_image_uri(counterfactual.get("removed_image"), "Removed Evidence")
    cf_state_uri = _slot_image_uri(counterfactual.get("counterfactual_image"), "Counterfactual State")
    shared_original_uri = _slot_image_uri(shared_focus.get("original_image"), "Original Image")
    shared_focus_uri = _slot_image_uri(shared_focus.get("shared_focus_image"), "Shared Focus Map")
    shared_disagreement_uri = _slot_image_uri(shared_focus.get("disagreement_image"), "Disagreement Map")

    metrics_curve_uri = (
        _build_metrics_curve_uri(curve_df) or _placeholder_data_uri("Metrics Curve", "Not available in this run.")
    )
    cf_curve_uri = (
        _build_counterfactual_curve_uri(progression_df)
        or _placeholder_data_uri("Confidence Trajectory", "Not available in this run.")
    )

    predicted_class = str(meta.get("predicted_class", "-"))
    confidence_pct = str(meta.get("confidence_pct", "-"))
    primary_method = str(meta.get("primary_method", "-"))
    runtime_s = str(meta.get("runtime_s", "-"))
    image_name = str(meta.get("image_name", "-"))
    method_set = str(meta.get("method_set", "-"))

    semantic_summary = str(semantic.get("summary", "")).strip() or "Not available in this run."
    semantic_note = "Τα semantic scores είναι CLIP-based δείκτες ομοιότητας και πρέπει να διαβάζονται ως υποστηρικτική ένδειξη."
    semantic_caption = str(semantic.get("focus_caption", "")).strip() or "Semantic focus region."

    counterfactual_summary = " ".join(_clean_lines(counterfactual.get("summary_lines", []))) if counterfactual else "Not available in this run."
    counterfactual_note = "Η counterfactual ανάλυση δείχνει πώς αλλάζει η πρόβλεψη όταν αφαιρείται σταδιακά η σημαντικότερη evidence περιοχή."

    shared_summary = " ".join(_clean_lines(shared_focus.get("summary_lines", []))) if shared_focus.get("available") else "Not available in this run."
    shared_note = (
        "Το shared focus δείχνει τις περιοχές όπου συμφωνούν πολλαπλοί explainers, "
        "ενώ το disagreement αναδεικνύει method-specific εστίαση."
    )

    prediction_snapshot_text = (
        f"Το μοντέλο προέβλεψε την κλάση {predicted_class} με βεβαιότητα {confidence_pct}. "
        f"Η κύρια οπτική εξήγηση παρήχθη με τη μέθοδο {primary_method} "
        f"και το τρέχον run περιλαμβάνει τη σύγκριση των εξής methods: {method_set}."
    )
    executive_summary = (
        f"Η παρούσα αναφορά συνοψίζει την explainability ανάλυση για την εικόνα {image_name}. "
        f"Η βασική πρόβλεψη ήταν {predicted_class} με confidence {confidence_pct}, "
        f"ενώ η κύρια μέθοδος ερμηνείας ήταν η {primary_method}. "
        f"{semantic_summary if semantic_summary != 'Not available in this run.' else 'Δεν παρήχθη semantic summary σε αυτό το run.'}"
    )

    sections = ["Overview", "Semantic", "Metrics", "Counterfactual", "Shared Focus"]
    section_availability = {
        "Overview": True,
        "Semantic": bool(semantic),
        "Metrics": bool(metrics),
        "Counterfactual": bool(counterfactual),
        "Shared Focus": bool(shared_focus.get("available")),
    }

    counterfactual_table_html = _df_to_html_table(progression_df, max_rows=9)
    if not progression_df.empty and len(progression_df) > 9:
        counterfactual_table_html += '<div class="note mt-4">Table truncated for readability.</div>'

    replacements = {
        "{{report_title}}": _escape(report_title),
        "{{generated_at}}": _escape(generated_at),
        "{{predicted_class}}": _escape(predicted_class),
        "{{confidence_pct}}": _escape(confidence_pct),
        "{{primary_method}}": _escape(primary_method),
        "{{runtime_s}}": _escape(runtime_s),
        "{{hero_image}}": overview_explained_uri,
        "{{executive_summary}}": _escape(executive_summary),
        "{{image_name}}": _escape(image_name),
        "{{method_set}}": _escape(method_set),
        "{{included_sections}}": _chips_html(sections, section_availability),
        "{{overview_original_image}}": overview_original_uri,
        "{{overview_explained_image}}": overview_explained_uri,
        "{{overview_heatmap_image}}": overview_heatmap_uri,
        "{{prediction_snapshot_text}}": _escape(prediction_snapshot_text),
        "{{top5_table}}": _df_to_html_table(top5_df, max_rows=5),
        "{{semantic_summary}}": _escape(semantic_summary),
        "{{semantic_note}}": _escape(semantic_note),
        "{{semantic_focus_image}}": semantic_focus_uri,
        "{{semantic_focus_caption}}": _escape(semantic_caption),
        "{{semantic_top_concepts_bars}}": _bars_html(_semantic_bar_items(semantic)),
        "{{semantic_contribution_table}}": _df_to_html_table(semantic_df, max_rows=12),
        "{{metrics_cards}}": _metric_cards_html(metrics),
        "{{metrics_curve_image}}": metrics_curve_uri,
        "{{metrics_details_table}}": _df_to_html_table(metrics_df, max_rows=14),
        "{{counterfactual_summary}}": _escape(counterfactual_summary),
        "{{counterfactual_note}}": _escape(counterfactual_note),
        "{{counterfactual_original_image}}": cf_original_uri,
        "{{counterfactual_removed_image}}": cf_removed_uri,
        "{{counterfactual_state_image}}": cf_state_uri,
        "{{counterfactual_curve_image}}": cf_curve_uri,
        "{{counterfactual_step_table}}": counterfactual_table_html,
        "{{shared_focus_summary}}": _escape(shared_summary),
        "{{shared_focus_note}}": _escape(shared_note),
        "{{shared_original_image}}": shared_original_uri,
        "{{shared_focus_image}}": shared_focus_uri,
        "{{shared_disagreement_image}}": shared_disagreement_uri,
        "{{shared_pairwise_table}}": _df_to_html_table(pairwise_df, max_rows=10),
        "{{shared_method_alignment_table}}": _df_to_html_table(method_df, max_rows=10),
    }

    template = f"""
<!DOCTYPE html>
<html lang="el">
<head>
  <meta charset="UTF-8" />
  <title>{{{{report_title}}}}</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 0;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 0;
      background: #e9e5dc;
      color: {TEXT};
      font-family: "DejaVu Sans", "Liberation Sans", Arial, Helvetica, sans-serif;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    .page {{
      width: 210mm;
      height: 297mm;
      margin: 0 auto;
      padding: 18mm 16mm 14mm 16mm;
      background: {BG};
      position: relative;
      page-break-after: always;
      overflow: hidden;
    }}

    .page:last-child {{
      page-break-after: auto;
    }}

    .top-rule {{
      width: 100%;
      height: 1.5mm;
      background: {TEAL};
      margin-bottom: 7mm;
    }}

    .brand-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12mm;
      color: {TEXT_SECONDARY};
      font-size: 8.5pt;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      color: {TEAL};
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    .brand-mark {{
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: {TEAL};
      color: {WHITE};
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 9pt;
      font-weight: 700;
    }}

    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 9mm;
      border-bottom: 1px solid #D8D1C6;
      padding-bottom: 5mm;
    }}

    .section-title-block {{
      max-width: 78%;
    }}

    h1, h2, h3 {{
      margin: 0;
      line-height: 1.12;
      color: {TEXT};
    }}

    h1 {{
      font-size: 31pt;
      letter-spacing: -0.03em;
      font-weight: 800;
      margin-bottom: 4mm;
    }}

    h2 {{
      font-size: 23pt;
      letter-spacing: -0.02em;
      font-weight: 800;
      margin-bottom: 2mm;
    }}

    h3 {{
      font-size: 12.5pt;
      font-weight: 800;
      margin-bottom: 2mm;
    }}

    .subtitle {{
      font-size: 10.5pt;
      color: {TEXT_SECONDARY};
      line-height: 1.45;
    }}

    .page-number {{
      font-size: 11pt;
      color: {COPPER};
      font-weight: 800;
      letter-spacing: 0.05em;
      padding-top: 1mm;
    }}

    .card {{
      background: {WHITE};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 5mm;
      position: relative;
      overflow: hidden;
    }}

    .card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1.5mm;
      background: {TEAL};
    }}

    .card.copper::before {{
      background: {COPPER};
    }}

    .card.gold::before {{
      background: {GOLD};
    }}

    .card.no-rule::before {{
      display: none;
    }}

    .card-title {{
      font-size: 12pt;
      font-weight: 800;
      margin: 1.5mm 0 2mm 0;
      color: {TEXT};
    }}

    .card-subtitle {{
      font-size: 8.2pt;
      color: {TEXT_SECONDARY};
      line-height: 1.35;
      margin-bottom: 3mm;
    }}

    .body-text {{
      font-size: 9pt;
      color: {TEXT};
      line-height: 1.55;
      white-space: pre-wrap;
    }}

    .muted {{
      color: {TEXT_SECONDARY};
    }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 4mm;
      margin-bottom: 6mm;
    }}

    .kpi-card {{
      background: {WHITE};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 4mm;
      min-height: 25mm;
    }}

    .kpi-label {{
      font-size: 7.5pt;
      color: {TEXT_SECONDARY};
      margin-bottom: 2.5mm;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-weight: 700;
    }}

    .kpi-value {{
      font-size: 14pt;
      font-weight: 800;
      color: {TEXT};
      line-height: 1.15;
    }}

    .kpi-hint {{
      font-size: 7.5pt;
      color: {TEXT_SECONDARY};
      margin-top: 2mm;
      line-height: 1.35;
    }}

    .hero-layout {{
      display: grid;
      grid-template-columns: 1.55fr 0.95fr;
      gap: 6mm;
      margin-bottom: 6mm;
    }}

    .hero-image {{
      width: 100%;
      height: 82mm;
      object-fit: contain;
      display: block;
      border-radius: 8px;
      background: {TABLE_ZEBRA};
    }}

    .image-card img,
    .chart-card img {{
      width: 100%;
      object-fit: contain;
      display: block;
      border-radius: 8px;
      background: {TABLE_ZEBRA};
    }}

    .image-card.tall img {{
      height: 70mm;
    }}

    .image-card.medium img {{
      height: 48mm;
    }}

    .image-card.small img {{
      height: 38mm;
    }}

    .chart-card img {{
      height: 62mm;
    }}

    .caption {{
      margin-top: 2mm;
      font-size: 7.8pt;
      color: {TEXT_SECONDARY};
      line-height: 1.35;
      text-align: center;
    }}

    .summary-box {{
      border-left: 3px solid {TEAL};
      padding-left: 4mm;
      margin-top: 3mm;
    }}

    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 2mm;
      margin-top: 3mm;
    }}

    .chip {{
      display: inline-block;
      border: 1px solid #D8D1C6;
      background: #FDFBF7;
      color: {TEXT};
      border-radius: 999px;
      padding: 1.5mm 3mm;
      font-size: 8pt;
      line-height: 1;
      white-space: nowrap;
    }}

    .chip.muted {{
      color: #8A8F93;
      background: #F3EEE6;
    }}

    .mini-divider {{
      width: 18mm;
      height: 1mm;
      background: {COPPER};
      margin: 4mm 0;
    }}

    .info-list {{
      display: grid;
      gap: 3mm;
      margin-top: 3mm;
      font-size: 9pt;
      line-height: 1.35;
    }}

    .info-row {{
      display: flex;
      justify-content: space-between;
      gap: 5mm;
      border-bottom: 1px solid #EEE8DE;
      padding-bottom: 2mm;
    }}

    .info-row strong {{
      color: {TEXT};
      text-align: right;
    }}

    .table-empty,
    .table-slot table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 8.4pt;
      line-height: 1.25;
    }}

    .table-empty {{
      padding: 7mm 4mm;
      text-align: center;
      color: {TEXT_SECONDARY};
      background: {WHITE};
      border: 1px dashed {BORDER};
      border-radius: 8px;
    }}

    .table-slot th,
    .table-slot td {{
      border: 1px solid {BORDER};
      padding: 2.1mm 2.5mm;
      text-align: left;
      vertical-align: middle;
    }}

    .table-slot th {{
      background: {TEAL};
      color: {WHITE};
      font-weight: 800;
    }}

    .table-slot tr:nth-child(even) td {{
      background: {TABLE_ZEBRA};
    }}

    .table-slot tr:nth-child(odd) td {{
      background: {WHITE};
    }}

    .table-slot.copper th {{
      background: {COPPER};
    }}

    .bars-slot {{
      width: 100%;
    }}

    .bars-slot .bar-row {{
      display: grid;
      grid-template-columns: 30mm 1fr 14mm;
      gap: 3mm;
      align-items: center;
      margin-bottom: 2.5mm;
      font-size: 8.4pt;
    }}

    .bars-slot .bar-track {{
      height: 4mm;
      background: #EFE9DE;
      border-radius: 999px;
      overflow: hidden;
    }}

    .bars-slot .bar-fill {{
      height: 100%;
      background: {TEAL};
      border-radius: 999px;
    }}

    .note {{
      background: #FDFBF7;
      border: 1px solid {BORDER};
      border-left: 3px solid {GOLD};
      border-radius: 8px;
      padding: 3mm 4mm;
      font-size: 8.1pt;
      color: {TEXT_SECONDARY};
      line-height: 1.45;
      margin-top: 3mm;
      white-space: pre-wrap;
    }}

    .footer {{
      position: absolute;
      left: 16mm;
      right: 16mm;
      bottom: 8mm;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 3mm;
      border-top: 1px solid #D8D1C6;
      font-size: 8pt;
      color: {TEXT_SECONDARY};
    }}

    .footer-page {{
      font-weight: 800;
      color: {TEAL};
      letter-spacing: 0.05em;
    }}

    .mt-4 {{ margin-top: 4mm; }}
    .mt-6 {{ margin-top: 6mm; }}
    .mt-8 {{ margin-top: 8mm; }}
    .mb-6 {{ margin-bottom: 6mm; }}

    .overview-images {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 5mm;
      margin-bottom: 6mm;
    }}

    .overview-bottom {{
      display: grid;
      grid-template-columns: 0.82fr 1.18fr;
      gap: 6mm;
    }}

    .semantic-layout {{
      display: grid;
      grid-template-columns: 0.9fr 1.1fr;
      gap: 6mm;
      margin-bottom: 6mm;
    }}

    .semantic-bottom {{
      display: grid;
      grid-template-columns: 0.82fr 1.18fr;
      gap: 6mm;
    }}

    .metrics-dashboard {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4mm;
      margin-bottom: 6mm;
    }}

    .counterfactual-images {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 5mm;
      margin-bottom: 6mm;
    }}

    .counterfactual-bottom {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 6mm;
    }}

    .shared-images {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 5mm;
      margin-bottom: 6mm;
    }}

    .shared-bottom {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6mm;
    }}

    @media print {{
      body {{
        background: {WHITE};
      }}
      .page {{
        margin: 0;
        box-shadow: none;
      }}
    }}
  </style>
</head>

<body>

  <section class="page">
    <div class="top-rule"></div>

    <div class="brand-row">
      <div class="brand">
        <span class="brand-mark">X</span>
        <span>XAI Thesis App</span>
      </div>
      <div>Generated: {{{{generated_at}}}}</div>
    </div>

    <h1>{{{{report_title}}}}</h1>
    <div class="subtitle">
      Αναφορά explainability για μία εικόνα με visual, metric και semantic evidence.
    </div>
    <div class="mini-divider"></div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Predicted Class</div>
        <div class="kpi-value">{{{{predicted_class}}}}</div>
        <div class="kpi-hint">Model output για την επιλεγμένη εικόνα.</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Confidence</div>
        <div class="kpi-value">{{{{confidence_pct}}}}</div>
        <div class="kpi-hint">Top prediction score.</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Primary Explainer</div>
        <div class="kpi-value">{{{{primary_method}}}}</div>
        <div class="kpi-hint">Main method used for analysis.</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Runtime</div>
        <div class="kpi-value">{{{{runtime_s}}}}</div>
        <div class="kpi-hint">Execution time for this run.</div>
      </div>
    </div>

    <div class="hero-layout">
      <div class="card image-card copper">
        <div class="card-title">Explained Visual Evidence</div>
        <div class="card-subtitle">{{{{primary_method}}}} explanation για την τρέχουσα πρόβλεψη.</div>
        <img class="hero-image" src="{{{{hero_image}}}}" alt="Hero explanation image" />
        <div class="caption">Κύρια οπτική εξήγηση που παρήχθη για το τρέχον run.</div>
      </div>

      <div class="card">
        <div class="card-title">Executive Summary</div>
        <div class="card-subtitle">Συνοπτική ανάγνωση του current XAI analysis.</div>
        <div class="body-text summary-box">
          {{{{executive_summary}}}}
        </div>

        <div class="info-list mt-6">
          <div class="info-row">
            <span class="muted">Image</span>
            <strong>{{{{image_name}}}}</strong>
          </div>
          <div class="info-row">
            <span class="muted">Method Set</span>
            <strong>{{{{method_set}}}}</strong>
          </div>
        </div>

        <div class="card-title mt-8">Included Sections</div>
        <div class="chips">
          {{{{included_sections}}}}
        </div>
      </div>
    </div>

    <div class="card gold">
      <div class="card-title">Report Notes</div>
      <div class="body-text">
        Η αναφορά αποτυπώνει την τρέχουσα κατάσταση ενός single-image run και προορίζεται ως ερμηνεύσιμο snapshot του app, όχι ως πλήρης dataset-level αξιολόγηση.
      </div>
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">01</span>
    </div>
  </section>


  <section class="page">
    <div class="section-header">
      <div class="section-title-block">
        <h2>Overview</h2>
        <div class="subtitle">
          Αρχική εικόνα, επιλεγμένη οπτική εξήγηση, raw heatmap και prediction context.
        </div>
      </div>
      <div class="page-number">02</div>
    </div>

    <div class="overview-images">
      <div class="card image-card medium copper">
        <div class="card-title">Original Image</div>
        <img src="{{{{overview_original_image}}}}" alt="Original image" />
        <div class="caption">Input image used for the current analysis.</div>
      </div>

      <div class="card image-card medium">
        <div class="card-title">Explained View</div>
        <img src="{{{{overview_explained_image}}}}" alt="Explained view" />
        <div class="caption">Overlay of the explanation on the original image.</div>
      </div>

      <div class="card image-card medium gold">
        <div class="card-title">Raw Heatmap</div>
        <img src="{{{{overview_heatmap_image}}}}" alt="Raw heatmap" />
        <div class="caption">Aggregated heatmap before contextual interpretation.</div>
      </div>
    </div>

    <div class="overview-bottom">
      <div class="card">
        <div class="card-title">Prediction Snapshot</div>
        <div class="card-subtitle">Σύντομο context για το selected model output.</div>
        <div class="body-text">
          {{{{prediction_snapshot_text}}}}
        </div>

        <div class="info-list mt-6">
          <div class="info-row">
            <span class="muted">Primary method</span>
            <strong>{{{{primary_method}}}}</strong>
          </div>
          <div class="info-row">
            <span class="muted">Predicted class</span>
            <strong>{{{{predicted_class}}}}</strong>
          </div>
          <div class="info-row">
            <span class="muted">Confidence</span>
            <strong>{{{{confidence_pct}}}}</strong>
          </div>
        </div>
      </div>

      <div class="card table-slot copper">
        <div class="card-title">Top-5 Predictions</div>
        <div class="card-subtitle">Highest model predictions for the current image.</div>
        {{{{top5_table}}}}
      </div>
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">02</span>
    </div>
  </section>


  <section class="page">
    <div class="section-header">
      <div class="section-title-block">
        <h2>Semantic Interpretation</h2>
        <div class="subtitle">
          Μετάφραση της οπτικής εστίασης σε υποστηρικτικά semantic concepts μέσω του semantic layer.
        </div>
      </div>
      <div class="page-number">03</div>
    </div>

    <div class="semantic-layout">
      <div class="card copper">
        <div class="card-title">Semantic Summary</div>
        <div class="card-subtitle">Human-readable interpretation of the focus region.</div>
        <div class="body-text">
          {{{{semantic_summary}}}}
        </div>

        <div class="note">
          {{{{semantic_note}}}}
        </div>
      </div>

      <div class="card image-card tall">
        <div class="card-title">Semantic Focus Region</div>
        <img src="{{{{semantic_focus_image}}}}" alt="Semantic focus region" />
        <div class="caption">{{{{semantic_focus_caption}}}}</div>
      </div>
    </div>

    <div class="semantic-bottom">
      <div class="card gold">
        <div class="card-title">Top Concepts</div>
        <div class="card-subtitle">Most relevant CLIP-based semantic indicators.</div>
        <div class="bars-slot">
          {{{{semantic_top_concepts_bars}}}}
        </div>
      </div>

      <div class="card table-slot copper">
        <div class="card-title">Concept Contribution Table</div>
        <div class="card-subtitle">Semantic score distribution across predefined concepts.</div>
        {{{{semantic_contribution_table}}}}
      </div>
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">03</span>
    </div>
  </section>


  <section class="page">
    <div class="section-header">
      <div class="section-title-block">
        <h2>Metrics</h2>
        <div class="subtitle">
          Per-image explanation indicators for the selected primary explainer.
        </div>
      </div>
      <div class="page-number">04</div>
    </div>

    <div class="card no-rule mb-6">
      <div class="card-title">Metric Snapshot</div>
      <div class="card-subtitle">Compact dashboard of the current explanation metrics.</div>
      <div class="metrics-dashboard">
        {{{{metrics_cards}}}}
      </div>
    </div>

    <div class="card chart-card">
      <div class="card-title">Deletion / Insertion Curves</div>
      <div class="card-subtitle">
        Faithfulness behavior as important regions are removed or restored.
      </div>
      <img src="{{{{metrics_curve_image}}}}" alt="Metrics curve" />
      <div class="caption">Deletion and insertion trajectories for the primary explanation.</div>
    </div>

    <div class="card table-slot copper mt-6">
      <div class="card-title">Metric Details</div>
      <div class="card-subtitle">Additional metric values available for this run.</div>
      {{{{metrics_details_table}}}}
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">04</span>
    </div>
  </section>


  <section class="page">
    <div class="section-header">
      <div class="section-title-block">
        <h2>Counterfactual</h2>
        <div class="subtitle">
          Τι συμβαίνει όταν αφαιρείται σταδιακά το πιο σημαντικό evidence.
        </div>
      </div>
      <div class="page-number">05</div>
    </div>

    <div class="card copper mb-6">
      <div class="card-title">Counterfactual Summary</div>
      <div class="card-subtitle">Short narrative of the counterfactual trajectory.</div>
      <div class="body-text">
        {{{{counterfactual_summary}}}}
      </div>
      <div class="note">
        {{{{counterfactual_note}}}}
      </div>
    </div>

    <div class="counterfactual-images">
      <div class="card image-card small copper">
        <div class="card-title">Original Image</div>
        <img src="{{{{counterfactual_original_image}}}}" alt="Counterfactual original image" />
        <div class="caption">Original input before evidence removal.</div>
      </div>

      <div class="card image-card small gold">
        <div class="card-title">Removed Evidence</div>
        <img src="{{{{counterfactual_removed_image}}}}" alt="Removed evidence image" />
        <div class="caption">Regions selected as important evidence.</div>
      </div>

      <div class="card image-card small">
        <div class="card-title">Counterfactual State</div>
        <img src="{{{{counterfactual_state_image}}}}" alt="Counterfactual state image" />
        <div class="caption">Image after removing selected evidence.</div>
      </div>
    </div>

    <div class="counterfactual-bottom">
      <div class="card chart-card">
        <div class="card-title">Confidence Trajectory</div>
        <div class="card-subtitle">Prediction confidence as evidence is removed.</div>
        <img src="{{{{counterfactual_curve_image}}}}" alt="Counterfactual confidence curve" />
        <div class="caption">Change in prediction score across removal steps.</div>
      </div>

      <div class="card table-slot copper">
        <div class="card-title">Step Table</div>
        <div class="card-subtitle">Key counterfactual removal steps.</div>
        {{{{counterfactual_step_table}}}}
      </div>
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">05</span>
    </div>
  </section>


  <section class="page">
    <div class="section-header">
      <div class="section-title-block">
        <h2>Shared Focus</h2>
        <div class="subtitle">
          Agreement and disagreement patterns between selected explainability methods.
        </div>
      </div>
      <div class="page-number">06</div>
    </div>

    <div class="card copper mb-6">
      <div class="card-title">Shared Focus Summary</div>
      <div class="card-subtitle">Interpretation of common and method-specific focus regions.</div>
      <div class="body-text">
        {{{{shared_focus_summary}}}}
      </div>
      <div class="note">
        {{{{shared_focus_note}}}}
      </div>
    </div>

    <div class="shared-images">
      <div class="card image-card small copper">
        <div class="card-title">Original Image</div>
        <img src="{{{{shared_original_image}}}}" alt="Shared focus original image" />
        <div class="caption">Input image used for method comparison.</div>
      </div>

      <div class="card image-card small">
        <div class="card-title">Shared Focus Map</div>
        <img src="{{{{shared_focus_image}}}}" alt="Shared focus map" />
        <div class="caption">Regions selected by multiple explainers.</div>
      </div>

      <div class="card image-card small gold">
        <div class="card-title">Disagreement Map</div>
        <img src="{{{{shared_disagreement_image}}}}" alt="Disagreement map" />
        <div class="caption">Regions where explainers differ most strongly.</div>
      </div>
    </div>

    <div class="shared-bottom">
      <div class="card table-slot copper">
        <div class="card-title">Pairwise Agreement</div>
        <div class="card-subtitle">Agreement indicators between explainer pairs.</div>
        {{{{shared_pairwise_table}}}}
      </div>

      <div class="card table-slot">
        <div class="card-title">Method Alignment</div>
        <div class="card-subtitle">Alignment of each method with the shared evidence region.</div>
        {{{{shared_method_alignment_table}}}}
      </div>
    </div>

    <div class="footer">
      <span>XAI Thesis App Report</span>
      <span class="footer-page">06</span>
    </div>
  </section>

</body>
</html>
"""

    html_report = template
    for placeholder, value in replacements.items():
        html_report = html_report.replace(placeholder, value)
    return html_report


def _find_chrome_binary() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _render_pdf_via_chrome(html_report: str) -> bytes | None:
    chrome_binary = _find_chrome_binary()
    if chrome_binary is None:
        return None

    with tempfile.TemporaryDirectory(prefix="xai-report-") as tmpdir:
        tmp_path = Path(tmpdir)
        html_path = tmp_path / "report.html"
        pdf_path = tmp_path / "report.pdf"
        html_path.write_text(html_report, encoding="utf-8")

        command = [
            chrome_binary,
            "--headless=new",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--allow-file-access-from-files",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=5000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        if completed.returncode != 0 or not pdf_path.exists():
            return None

        return pdf_path.read_bytes()


def _fallback_draw_table(ax: plt.Axes, df: pd.DataFrame, max_rows: int = 12) -> None:
    ax.axis("off")
    display_df = df.copy().head(max_rows).fillna("")
    for column in display_df.columns:
        display_df[column] = display_df[column].map(lambda value: str(value))
    if display_df.empty:
        ax.text(0.5, 0.5, "Not available in this run.", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=TEXT_SECONDARY)
        return
    table = ax.table(
        cellText=display_df.values,
        colLabels=[str(col) for col in display_df.columns],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.3)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.5)
        if row_idx == 0:
            cell.set_facecolor(TEAL)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor(TABLE_ZEBRA)


def _render_fallback_pdf(payload: Mapping[str, Any]) -> bytes:
    report_title = str(payload.get("title", "XAI Analysis Report"))
    meta = dict(payload.get("meta", {}))
    overview = dict(payload.get("overview", {}))
    semantic = dict(payload.get("semantic", {}))
    metrics = dict(payload.get("metrics", {}))
    counterfactual = dict(payload.get("counterfactual", {}))
    shared_focus = dict(payload.get("shared_focus", {}))

    pages: list[tuple[str, list[tuple[str, Any]], list[pd.DataFrame]]] = [
        (
            "Executive Summary",
            [
                ("Predicted class", meta.get("predicted_class", "-")),
                ("Confidence", meta.get("confidence_pct", "-")),
                ("Primary explainer", meta.get("primary_method", "-")),
                ("Runtime", meta.get("runtime_s", "-")),
            ],
            [overview.get("top5_df") if isinstance(overview.get("top5_df"), pd.DataFrame) else pd.DataFrame()],
        ),
        (
            "Overview",
            [("Image", meta.get("image_name", "-")), ("Method set", meta.get("method_set", "-"))],
            [overview.get("top5_df") if isinstance(overview.get("top5_df"), pd.DataFrame) else pd.DataFrame()],
        ),
        (
            "Semantic Interpretation",
            [("Summary", semantic.get("summary", "Not available in this run."))],
            [semantic.get("score_table") if isinstance(semantic.get("score_table"), pd.DataFrame) else pd.DataFrame()],
        ),
        (
            "Metrics",
            [
                ("Deletion AUC", metrics.get("deletion_auc", "-")),
                ("Insertion AUC", metrics.get("insertion_auc", "-")),
                ("Robustness", metrics.get("robustness", "-")),
            ],
            [metrics.get("details_df") if isinstance(metrics.get("details_df"), pd.DataFrame) else pd.DataFrame()],
        ),
        (
            "Counterfactual",
            [("Summary", " ".join(_clean_lines(counterfactual.get("summary_lines", []))) if counterfactual else "Not available in this run.")],
            [counterfactual.get("progression_df") if isinstance(counterfactual.get("progression_df"), pd.DataFrame) else pd.DataFrame()],
        ),
        (
            "Shared Focus",
            [("Summary", " ".join(_clean_lines(shared_focus.get("summary_lines", []))) if shared_focus.get("available") else "Not available in this run.")],
            [
                shared_focus.get("pairwise_df") if isinstance(shared_focus.get("pairwise_df"), pd.DataFrame) else pd.DataFrame(),
                shared_focus.get("method_df") if isinstance(shared_focus.get("method_df"), pd.DataFrame) else pd.DataFrame(),
            ],
        ),
    ]

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for page_num, (title, rows, tables) in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(WHITE)
            fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold", color=TEXT, ha="left", va="top")
            fig.text(0.93, 0.95, f"{page_num:02d}", fontsize=10, fontweight="bold", color=TEAL, ha="right", va="top")
            fig.add_artist(plt.Line2D([0.07, 0.93], [0.925, 0.925], color=TEAL, linewidth=2.0, transform=fig.transFigure))

            info_text = "\n".join(f"{label}: {value}" for label, value in rows)
            fig.text(0.07, 0.86, _wrap_text(info_text, width=85), fontsize=10.2, color=TEXT, ha="left", va="top", linespacing=1.55)

            top = 0.54
            for index, table_df in enumerate(tables):
                ax = fig.add_axes([0.07, top - index * 0.28, 0.86, 0.20])
                _fallback_draw_table(ax, table_df, max_rows=12)

            fig.text(0.07, 0.03, "XAI Thesis App Report", fontsize=8.4, color=TEXT_SECONDARY, ha="left", va="bottom")
            fig.text(0.93, 0.03, f"{page_num:02d}", fontsize=9, color=TEAL, ha="right", va="bottom", fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.12, facecolor=fig.get_facecolor())
            plt.close(fig)
    return buffer.getvalue()


def build_pdf_report(payload: Mapping[str, Any]) -> bytes:
    html_report = _build_html_report(payload)
    browser_pdf = _render_pdf_via_chrome(html_report)
    if browser_pdf is not None:
        return browser_pdf
    return _render_fallback_pdf(payload)
