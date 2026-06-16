"""
src/components/pdf_reporter.py
================================
Generates a clean, perfectly-aligned A4 PDF ESG report.

Design rules
------------
- Title: "ESG Report Analysis" — no colour, plain black Helvetica-Bold
- Fixed A4 content width = 170mm (A4 210mm - 20mm margins each side)
- ALL columns set as explicit fractions of CONTENT_W — nothing overflows
- No negative margins anywhere
- align-items: flex-start equivalent — all text left-aligned except explicit centres
- Charts embedded as PNG bytes at exact CONTENT_W width
- Every table uses colWidths that sum to exactly CONTENT_W
- Spacer heights are fixed constants, never calculated mid-flow
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image, PageBreak,
)

if TYPE_CHECKING:
    from src.pipeline.predict_pipeline import PipelineResult

# ─── Page geometry (all in points) ───────────────────────────────────────────
PAGE_W, PAGE_H = A4                       # 595.28 × 841.89 pt
LM = RM = 20 * mm                        # 20 mm margins
TM = BM = 18 * mm
CW = PAGE_W - LM - RM                    # 155.28 mm ≈ 439.37 pt  (CONTENT WIDTH)

# ─── Palette ──────────────────────────────────────────────────────────────────
C_BLACK      = colors.HexColor("#111827")
C_DARKGRAY   = colors.HexColor("#374151")
C_MIDGRAY    = colors.HexColor("#6B7280")
C_LIGHTGRAY  = colors.HexColor("#9CA3AF")
C_BORDER     = colors.HexColor("#E5E7EB")
C_STRIPLIGHT = colors.HexColor("#F9FAFB")
C_GREEN_DARK = colors.HexColor("#14532D")
C_GREEN_MID  = colors.HexColor("#16A34A")
C_GREEN_LITE = colors.HexColor("#DCFCE7")
C_GREEN_BG   = colors.HexColor("#F0FDF4")
C_AMBER      = colors.HexColor("#D97706")
C_AMBER_LITE = colors.HexColor("#FEF3C7")
C_RED        = colors.HexColor("#DC2626")
C_RED_LITE   = colors.HexColor("#FEE2E2")
C_BLUE       = colors.HexColor("#1D4ED8")
C_PURPLE     = colors.HexColor("#6D28D9")
C_WHITE      = colors.white

def _sc(v: float):
    if v >= 75: return C_GREEN_MID
    if v >= 50: return C_AMBER
    return C_RED

def _sc_hex(v: float) -> str:
    if v >= 75: return "#16A34A"
    if v >= 50: return "#D97706"
    return "#DC2626"

def _grade(s: float) -> str:
    if s >= 85: return "A+"
    if s >= 75: return "A"
    if s >= 65: return "B+"
    if s >= 55: return "B"
    if s >= 45: return "C"
    return "D"

def _cls_label(cls: str) -> str:
    return {"SEBI_BRSR": "SEBI BRSR Report",
            "SUSTAINABILITY_REPORT": "Sustainability Report",
            "INVALID_DOCUMENT": "Invalid Document"}.get(cls, cls)

def _perf(s: float) -> str:
    if s >= 75: return "strong"
    if s >= 55: return "moderate"
    return "needs improvement"

# ─── Style factory ────────────────────────────────────────────────────────────
def _S(fname="Helvetica", size=10, color=C_BLACK, align=TA_LEFT,
       leading=None, bold=False, space_before=0, space_after=4):
    if bold and "Bold" not in fname:
        fname = fname + "-Bold"
    return ParagraphStyle(
        f"s_{fname}_{size}_{id(color)}",
        fontName=fname,
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=leading or size * 1.4,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )

# ─── Chart generators ─────────────────────────────────────────────────────────

def _hbar_png(metrics: dict, title: str) -> bytes:
    """Horizontal bar chart — fixed 6.5 × variable height, 160 dpi."""
    n      = len(metrics)
    height = max(2.4, n * 0.52 + 0.6)
    fig, ax = plt.subplots(figsize=(6.5, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = list(metrics.keys())
    values = list(metrics.values())
    y      = np.arange(n)
    cols   = [_sc_hex(v) for v in values]

    bars = ax.barh(y, values, height=0.55, color=cols, edgecolor="none", zorder=3)

    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 102), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=_sc_hex(val))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color="#111827")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Score / 100", fontsize=8, color="#6B7280")
    ax.tick_params(axis="x", labelsize=8, colors="#9CA3AF")
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=10, fontweight="bold",
                 color="#111827", loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#F3F4F6", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _vbar_png(e: float, s: float, g: float) -> bytes:
    """Vertical grouped pillar bar — fixed 5.0 × 2.8 inches."""
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pillars = ["Environment", "Social", "Governance"]
    values  = [e, s, g]
    cols    = [_sc_hex(v) for v in values]
    x       = np.arange(3)

    bars = ax.bar(x, values, width=0.48, color=cols, edgecolor="none", zorder=3)
    for bar, val, col in zip(bars, values, cols):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col)

    ax.axhline(75, color="#D1D5DB", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(2.38, 76.5, "Target 75", fontsize=7.5, color="#9CA3AF", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(pillars, fontsize=10, color="#111827", fontweight="500")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score / 100", fontsize=8, color="#6B7280")
    ax.tick_params(axis="y", labelsize=8, colors="#9CA3AF")
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#F3F4F6", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Table style builder ──────────────────────────────────────────────────────

def _tbl(hdr_color=C_GREEN_DARK, stripe=True,
         extra_cmds: list | None = None) -> TableStyle:
    cmds = [
        ("BACKGROUND",    (0, 0), (-1,  0), hdr_color),
        ("TEXTCOLOR",     (0, 0), (-1,  0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1,  0), 8),
        ("ALIGN",         (0, 0), (-1,  0), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C_DARKGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.4, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
    ]
    if stripe:
        cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [C_WHITE, C_STRIPLIGHT]))
    if extra_cmds:
        cmds.extend(extra_cmds)
    return TableStyle(cmds)


# ─── Analysis text ────────────────────────────────────────────────────────────

def _analysis(r: "PipelineResult") -> dict:
    sc   = r.scores
    clf  = r.inference
    anon = r.anonymization
    ing  = r.ingestion

    summary = (
        f"This document was classified as a <b>{_cls_label(clf.predicted_class)}</b> "
        f"with {clf.confidence*100:.1f}% model confidence. "
        f"The pipeline processed {ing.char_count:,} characters "
        f"({'across ' + str(ing.page_count) + ' pages' if ing.page_count else 'from text input'}), "
        f"completing in {r.duration_ms:.0f} ms. "
        f"The overall ESG performance score is <b>{sc.overall_score}/100</b> "
        f"(Grade <b>{sc.grade}</b>), reflecting {_perf(sc.overall_score)} ESG disclosure maturity. "
        f"Scores reflect disclosure depth across 16 sub-metrics: "
        f"5 Environmental (weight 40%), 6 Social (weight 35%), and 5 Governance (weight 25%)."
    )

    top_e  = max(sc.e_metrics, key=sc.e_metrics.get)
    weak_e = min(sc.e_metrics, key=sc.e_metrics.get)
    env = (
        f"The Environmental pillar scores <b>{sc.environment_score}/100</b> "
        f"({_perf(sc.environment_score)}). "
        f"<b>{top_e}</b> is the strongest metric at {sc.e_metrics[top_e]:.0f}/100. "
        f"<b>{weak_e}</b> scores {sc.e_metrics[weak_e]:.0f}/100 and is the primary "
        f"opportunity to deepen environmental reporting. "
        f"Best-practice disclosures include GHG Scope 1/2/3 inventory with external assurance, "
        f"energy transition targets, water stewardship programmes, waste circularity metrics, "
        f"and biodiversity impact assessments."
    )

    top_s  = max(sc.s_metrics, key=sc.s_metrics.get)
    weak_s = min(sc.s_metrics, key=sc.s_metrics.get)
    soc = (
        f"The Social pillar scores <b>{sc.social_score}/100</b> "
        f"({_perf(sc.social_score)}). "
        f"<b>{top_s}</b> leads at {sc.s_metrics[top_s]:.0f}/100. "
        f"<b>{weak_s}</b> scores {sc.s_metrics[weak_s]:.0f}/100 and warrants deeper disclosure. "
        f"Strong social reporting covers employee wellbeing benefits (health, maternity, PF/ESI), "
        f"occupational health and safety with LTIFR trends, human rights due diligence "
        f"across the value chain, diversity and inclusion metrics, and structured CSR programmes."
    )

    top_g  = max(sc.g_metrics, key=sc.g_metrics.get)
    weak_g = min(sc.g_metrics, key=sc.g_metrics.get)
    gov = (
        f"The Governance pillar scores <b>{sc.governance_score}/100</b> "
        f"({_perf(sc.governance_score)}). "
        f"<b>{top_g}</b> leads at {sc.g_metrics[top_g]:.0f}/100. "
        f"<b>{weak_g}</b> scores {sc.g_metrics[weak_g]:.0f}/100, flagging a potential gap. "
        f"Investors and regulators assess board independence, anti-corruption frameworks, "
        f"statutory compliance monitoring, enterprise risk management, and material disclosures."
    )

    pii = (
        f"The anonymization engine performed <b>{anon.total_redactions} redactions</b> "
        f"across {len(anon.redaction_counts)} entity categories using a dual-pass pipeline: "
        f"Pass A (linguistic NER — honorific-anchored person detection and corporate-suffix "
        f"ORG detection) and Pass B (nine deterministic regex patterns: CIN, PAN, financials, "
        f"email, phone, IBAN, bank accounts, addresses, and PIN codes)."
    )

    recs = []
    if sc.environment_score < 75:
        recs.append("Expand environmental disclosures — prioritise GHG Scope 3 inventory and water intensity reporting.")
    if sc.social_score < 75:
        recs.append("Strengthen social reporting — include LTIFR trends, gender pay parity, and human rights due diligence.")
    if sc.governance_score < 75:
        recs.append("Improve governance transparency — detail board committee composition and risk management frameworks.")
    if sc.overall_score >= 75:
        recs.append("Pursue third-party reasonable assurance for BRSR Core KPIs to elevate report credibility.")
    recs.append("Align all disclosures with GRI Universal Standards and SEBI BRSR Core mandatory indicators.")
    recs.append("Conduct a formal double materiality assessment to identify and prioritise ESG topics.")

    return dict(summary=summary, environment=env, social=soc,
                governance=gov, pii=pii, recommendations=recs)


# ─── Section header helper ────────────────────────────────────────────────────

def _sec(title: str, bar_color=C_GREEN_DARK) -> list:
    return [
        Paragraph(title, _S(size=13, bold=True, space_before=4, space_after=3)),
        HRFlowable(width=CW, thickness=1.5, color=bar_color,
                   spaceAfter=6, spaceBefore=0),
    ]


# ─── Main builder ─────────────────────────────────────────────────────────────

def generate_pdf(result: "PipelineResult") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM,
        title="ESG Report Analysis",
        author="ESG Document Intelligence",
    )

    sc   = result.scores
    clf  = result.inference
    anon = result.anonymization
    ing  = result.ingestion
    txt  = _analysis(result)
    now  = datetime.now().strftime("%d %B %Y, %H:%M")
    story = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═══════════════════════════════════════════════════════════════════════════

    # Title — plain black, no colour
    story.append(Paragraph(
        "ESG Report Analysis",
        _S(size=26, bold=True, color=C_BLACK, space_before=0, space_after=2),
    ))
    story.append(HRFlowable(width=CW, thickness=2, color=C_BORDER,
                             spaceAfter=10, spaceBefore=0))

    # Meta row — 3 columns, explicit widths summing to CW
    c1 = round(CW * 0.40)
    c2 = round(CW * 0.35)
    c3 = CW - c1 - c2
    meta_rows = [[
        Paragraph(f"<b>Generated:</b> {now}",       _S(size=8, color=C_MIDGRAY)),
        Paragraph(f"<b>Source:</b> {ing.source}",   _S(size=8, color=C_MIDGRAY)),
        Paragraph(f"<b>Duration:</b> {result.duration_ms:.0f} ms",
                  _S(size=8, color=C_MIDGRAY)),
    ]]
    meta_tbl = Table(meta_rows, colWidths=[c1, c2, c3])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_STRIPLIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.4, C_BORDER),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 12))

    # ── Score summary table (4 columns, exact widths) ─────────────────────────
    w4 = [round(CW * f) for f in [0.30, 0.18, 0.18, 0.34]]
    w4[-1] = CW - sum(w4[:-1])   # fix rounding

    def _score_cell(label, score, weight, extra=""):
        return [
            Paragraph(label,  _S(size=8.5, bold=True)),
            Paragraph(f"{score}/100", _S(size=8.5, color=_sc(score), bold=True, align=TA_CENTER)),
            Paragraph(weight, _S(size=8.5, color=C_MIDGRAY, align=TA_CENTER)),
            Paragraph(extra or ("Strong" if score >= 75 else "Moderate" if score >= 55 else "Needs work"),
                      _S(size=8.5, align=TA_CENTER)),
        ]

    score_rows = [
        [Paragraph(h, _S(size=8.5, bold=True, color=C_WHITE))
         for h in ["Pillar", "Score", "Weight", "Status"]],
        _score_cell("Environment",  sc.environment_score, "40%"),
        _score_cell("Social",       sc.social_score,      "35%"),
        _score_cell("Governance",   sc.governance_score,  "25%"),
        [Paragraph("<b>Overall</b>", _S(size=8.5, bold=True)),
         Paragraph(f"<b>{sc.overall_score}/100</b>",
                   _S(size=8.5, bold=True, color=_sc(sc.overall_score), align=TA_CENTER)),
         Paragraph("100%", _S(size=8.5, color=C_MIDGRAY, align=TA_CENTER)),
         Paragraph(f"<b>Grade {sc.grade}</b>",
                   _S(size=8.5, bold=True, align=TA_CENTER))],
    ]
    score_tbl = Table(score_rows, colWidths=w4)
    score_tbl.setStyle(_tbl(extra_cmds=[
        ("BACKGROUND", (0, 4), (-1, 4), C_GREEN_BG),
        ("FONTNAME",   (0, 4), (-1, 4), "Helvetica-Bold"),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 14))

    # ── Classification badge row ──────────────────────────────────────────────
    cls_color = {
        "SEBI_BRSR": C_GREEN_DARK,
        "SUSTAINABILITY_REPORT": C_BLUE,
    }.get(clf.predicted_class, C_RED)

    badge_data = [[
        Paragraph(
            f"<b>Document Type:</b> {_cls_label(clf.predicted_class)}"
            f" &nbsp;&nbsp; <b>Confidence:</b> {clf.confidence*100:.1f}%"
            f" &nbsp;&nbsp; <b>Overall Score:</b> {sc.overall_score}/100"
            f" &nbsp;&nbsp; <b>Grade:</b> {sc.grade}",
            _S(size=9.5, bold=True, color=C_WHITE),
        )
    ]]
    badge_tbl = Table(badge_data, colWidths=[CW])
    badge_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), cls_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(badge_tbl)
    story.append(Spacer(1, 14))

    # ── Pillar vertical bar chart ─────────────────────────────────────────────
    vbar_bytes = _vbar_png(sc.environment_score, sc.social_score, sc.governance_score)
    # Compute height preserving aspect ratio within CW
    vbar_w = CW
    vbar_h = CW * (2.8 / 5.0)
    story.append(Image(io.BytesIO(vbar_bytes), width=vbar_w, height=vbar_h))
    story.append(Paragraph(
        "Figure 1: ESG pillar scores (target line at 75/100)",
        _S(size=8, color=C_LIGHTGRAY, align=TA_CENTER, space_after=2),
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    story += _sec("Executive Summary")
    story.append(Paragraph(txt["summary"], _S(size=9.5, leading=15, space_after=8)))
    story.append(Spacer(1, 8))

    # ── Detailed score table (all 16 metrics) ─────────────────────────────────
    story += _sec("Full Metric Scorecard", bar_color=C_MIDGRAY)

    wa = [round(CW * f) for f in [0.36, 0.16, 0.12, 0.36]]
    wa[-1] = CW - sum(wa[:-1])

    all_metrics_rows = [
        [Paragraph(h, _S(size=8, bold=True, color=C_WHITE))
         for h in ["Metric", "Score", "Pillar", "Assessment"]],
    ]
    pillar_map = {
        **{k: "Environmental" for k in sc.e_metrics},
        **{k: "Social"        for k in sc.s_metrics},
        **{k: "Governance"    for k in sc.g_metrics},
    }
    all_m = {**sc.e_metrics, **sc.s_metrics, **sc.g_metrics}
    for metric, score in sorted(all_m.items(), key=lambda x: -x[1]):
        assessment = ("Excellent — strong disclosure" if score >= 75 else
                      "Adequate — room for improvement" if score >= 50 else
                      "Weak — needs significant enhancement")
        all_metrics_rows.append([
            Paragraph(metric,            _S(size=8)),
            Paragraph(f"{score:.0f}/100",_S(size=8, color=_sc(score), bold=True, align=TA_CENTER)),
            Paragraph(pillar_map[metric],_S(size=8, color=C_MIDGRAY, align=TA_CENTER)),
            Paragraph(assessment,        _S(size=8)),
        ])

    all_m_tbl = Table(all_metrics_rows, colWidths=wa)
    all_m_tbl.setStyle(_tbl(extra_cmds=[
        ("ALIGN", (1, 0), (2, -1), "CENTER"),
    ]))
    story.append(all_m_tbl)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — ENVIRONMENTAL
    # ═══════════════════════════════════════════════════════════════════════════
    story += _sec("Environmental Performance", bar_color=C_GREEN_MID)
    story.append(Paragraph(txt["environment"], _S(size=9.5, leading=15, space_after=10)))

    e_img = _hbar_png(sc.e_metrics, "Environmental Sub-Metrics")
    e_h   = CW * (max(2.4, len(sc.e_metrics) * 0.52 + 0.6) / 6.5)
    story.append(Image(io.BytesIO(e_img), width=CW, height=e_h))
    story.append(Paragraph(
        "Figure 2: Environmental sub-metric scores (out of 100)",
        _S(size=8, color=C_LIGHTGRAY, align=TA_CENTER),
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — SOCIAL
    # ═══════════════════════════════════════════════════════════════════════════
    story += _sec("Social Performance", bar_color=C_BLUE)
    story.append(Paragraph(txt["social"], _S(size=9.5, leading=15, space_after=10)))

    s_img = _hbar_png(sc.s_metrics, "Social Sub-Metrics")
    s_h   = CW * (max(2.4, len(sc.s_metrics) * 0.52 + 0.6) / 6.5)
    story.append(Image(io.BytesIO(s_img), width=CW, height=s_h))
    story.append(Paragraph(
        "Figure 3: Social sub-metric scores (out of 100)",
        _S(size=8, color=C_LIGHTGRAY, align=TA_CENTER),
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — GOVERNANCE
    # ═══════════════════════════════════════════════════════════════════════════
    story += _sec("Governance Performance", bar_color=C_PURPLE)
    story.append(Paragraph(txt["governance"], _S(size=9.5, leading=15, space_after=10)))

    g_img = _hbar_png(sc.g_metrics, "Governance Sub-Metrics")
    g_h   = CW * (max(2.4, len(sc.g_metrics) * 0.52 + 0.6) / 6.5)
    story.append(Image(io.BytesIO(g_img), width=CW, height=g_h))
    story.append(Paragraph(
        "Figure 4: Governance sub-metric scores (out of 100)",
        _S(size=8, color=C_LIGHTGRAY, align=TA_CENTER),
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — PII + RECOMMENDATIONS + METHODOLOGY
    # ═══════════════════════════════════════════════════════════════════════════
    story += _sec("PII Anonymization Report", bar_color=C_AMBER)
    story.append(Paragraph(txt["pii"], _S(size=9.5, leading=15, space_after=8)))

    if anon.redaction_counts:
        desc_map = {
            "[REDACTED_PERSON]":     "Executive / director names via honorific NER",
            "[REDACTED_ORG]":        "Company / firm names via corporate suffix NER",
            "[REDACTED_FINANCIALS]": "Monetary values — INR, USD, crores, lakhs, etc.",
            "[REDACTED_EMAIL]":      "Email addresses",
            "[REDACTED_PHONE]":      "Phone numbers (Indian and international)",
            "[REDACTED_CIN]":        "Corporate Identity Number (MCA format)",
            "[REDACTED_PAN]":        "Permanent Account Number",
            "[REDACTED_ACCOUNT]":    "Bank account and IBAN numbers",
            "[REDACTED_ADDRESS]":    "Postal addresses and 6-digit PIN codes",
        }
        wp = [round(CW * f) for f in [0.22, 0.10, 0.18, 0.50]]
        wp[-1] = CW - sum(wp[:-1])

        pii_rows = [[
            Paragraph(h, _S(size=8, bold=True, color=C_WHITE))
            for h in ["Entity Type", "Count", "Method", "Description"]
        ]]
        for tag, count in sorted(anon.redaction_counts.items(), key=lambda x: -x[1]):
            entity = tag.replace("[REDACTED_", "").replace("]", "")
            method = "NER — Pass A" if tag in ("[REDACTED_PERSON]", "[REDACTED_ORG]") else "Regex — Pass B"
            pii_rows.append([
                Paragraph(entity,  _S(size=8)),
                Paragraph(str(count), _S(size=8, align=TA_CENTER, bold=True)),
                Paragraph(method,  _S(size=8, color=C_MIDGRAY)),
                Paragraph(desc_map.get(tag, "Sensitive entity"), _S(size=8)),
            ])
        pii_rows.append([
            Paragraph("<b>TOTAL</b>", _S(size=8, bold=True)),
            Paragraph(f"<b>{anon.total_redactions}</b>",
                      _S(size=8, bold=True, align=TA_CENTER)),
            Paragraph("", _S(size=8)),
            Paragraph("", _S(size=8)),
        ])

        pii_tbl = Table(pii_rows, colWidths=wp)
        pii_tbl.setStyle(_tbl(
            hdr_color=colors.HexColor("#92400E"),
            extra_cmds=[
                ("ALIGN",      (1, 0), (1, -1), "CENTER"),
                ("BACKGROUND", (0, len(pii_rows)-1), (-1, -1), C_AMBER_LITE),
                ("FONTNAME",   (0, len(pii_rows)-1), (-1, -1), "Helvetica-Bold"),
            ],
        ))
        story.append(pii_tbl)

    story.append(Spacer(1, 14))

    # ── Recommendations ───────────────────────────────────────────────────────
    story += _sec("Recommendations", bar_color=C_RED)
    for i, rec in enumerate(txt["recommendations"], 1):
        story.append(Paragraph(
            f"<b>{i}.</b>&nbsp; {rec}",
            _S(size=9.5, leading=14, space_after=5),
        ))

    story.append(Spacer(1, 14))

    # ── Methodology ───────────────────────────────────────────────────────────
    story += _sec("Methodology", bar_color=C_MIDGRAY)
    story.append(Paragraph(
        "This report was produced by the ESG Document Classification & Anonymization "
        "Microservice. <b>Classification</b> uses a multi-scale Keras Text-CNN "
        "(parallel Conv1D at 2, 3, 4-gram widths, SpatialDropout1D, hybrid GlobalMax "
        "+ GlobalAverage pooling, L2 regularization, BatchNormalization) trained on "
        "900 synthetic domain-specific samples with disjoint train/validation pools. "
        "<b>ESG Scoring</b> uses weighted keyword matching across 16 sub-metrics with "
        "dynamic per-metric target calibration; weights: E×40%, S×35%, G×25%. "
        "<b>Anonymization</b> uses a dual-pass pipeline: Pass A (linguistic NER) and "
        "Pass B (nine deterministic regex patterns with flag isolation and "
        "case-insensitive matching). Scores reflect disclosure depth, not absolute ESG performance.",
        _S(size=8.5, leading=13, color=C_MIDGRAY, space_after=4),
    ))

    doc.build(story)
    return buf.getvalue()
