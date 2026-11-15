"""Shared style for the mass-balance main and supplementary figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "supplementary_completion" / "tables"
FIGURES = ROOT / "supplementary_completion" / "figures"
WIDTH = 7.0
BASE, SMALL, TITLE, LETTER = 8.0, 7.0, 9.0, 10.0
INK, MUTED, GRID, BAND, BOX = "#263541", "#586874", "#E5EBEF", "#EFF3F5", "#F1F3F4"
AXES, REFERENCE = "#647581", "#7D8B96"
BASELINE_GREY, LOSS, LOSS_FILL, CONTROL = "#7C8993", "#D55E00", "#FBEFE6", "#475763"
BASELINE_DASH = (0, (4, 2.4))
MODELS = [("GraphConv", "GraphConv"), ("Chemprop", "Chemprop"), ("ChemBERTa_head", "Frozen ChemBERTa + MLP")]
ENDPOINTS = {"t_half": ("Half-life", r"$t_{1/2}$"), "Vd": ("Volume of distribution", r"$V_d$"),
             "CL": ("Clearance", r"$CL$"), "Cmax": ("Peak concentration", r"$C_{\max}$"),
             "Tmax": ("Time to peak", r"$T_{\max}$"), "F": ("Bioavailability", r"$F$"),
             "PPB": ("Protein binding", r"$PPB$")}
ENDPOINT_COLOURS = {"t_half": "#B2182B", "Vd": "#00865A", "CL": "#9A7B00", "Cmax": "#C04B8B",
                    "Tmax": "#596B2F", "F": "#238C8C", "PPB": "#6B4530"}
ENDPOINT_UNIT = {name: ("percentage_point" if name in ("F", "PPB") else "log10") for name in ENDPOINTS}


def configure():
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        family, math = "Arial", {"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                                 "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"}
    except ValueError:
        family, math = "DejaVu Sans", {"mathtext.fontset": "dejavusans"}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"], "font.size": BASE,
        "axes.labelsize": BASE, "axes.titlesize": TITLE, "axes.labelcolor": INK, "text.color": INK,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "xtick.color": INK, "ytick.color": INK,
        "legend.fontsize": SMALL, "legend.frameon": False, "axes.edgecolor": AXES, "axes.linewidth": .6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": .6, "ytick.major.width": .6, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white", **math})
    return family


def letter(fig, text, x, y):
    fig.text(x, y, text, fontsize=LETTER, fontweight="bold", color=INK, va="top", ha="left")


def model_title(ax, title):
    ax.set_title(title.replace("ChemBERTa + MLP", "ChemBERTa\n+ MLP"), fontweight="bold", pad=4,
                 va="bottom")


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    problems = []
    for i, (a, ba) in enumerate(texts):
        if not (fig.bbox.x0 <= ba.x0 and ba.x1 <= fig.bbox.x1 and fig.bbox.y0 <= ba.y0 and ba.y1 <= fig.bbox.y1):
            problems.append(("outside canvas", a.get_text()))
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                problems.append(("overlap", a.get_text(), b.get_text()))
    return problems


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300)
    problems = check_layout(fig)
    plt.close(fig)
    print(stem, "layout problems:", problems)
    return problems
