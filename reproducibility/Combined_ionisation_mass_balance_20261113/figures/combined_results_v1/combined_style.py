"""Shared presentation settings for the combined study figures."""

from matplotlib import rcParams


WIDTH = 7.0
LEFT, RIGHT, WSPACE = .255, .978, .18
HEADING_X = .035
TITLE_X = .072
INK = "#263541"
MUTED = "#586874"
GRID = "#E5EBEF"
AXES = "#647581"
BOX = "#F1F3F4"
BAND = "#EFF3F5"
CONTROL = "#475763"
GREY, BLUE, ORANGE, PURPLE = "#7C8993", "#0072B2", "#D55E00", "#7B4AB2"
BLUE_FILL, ORANGE_FILL = "#E6F1F7", "#FBEFE6"
CATEGORIES = (
    ("Acid-only", "#333333", "o"),
    ("Base-only", "#666666", "^"),
    ("Neither-active", "#999999", "s"),
    ("Dual-active", "#BBBBBB", "D"),
)
ENDPOINT_COLORS = {
    "t_half": "#B2182B",
    "Vd": "#00865A",
    "CL": "#9A7B00",
    "Cmax": "#C04B8B",
    "Tmax": "#596B2F",
    "F": "#238C8C",
    "PPB": "#6B4530",
}
CONDITIONS = {
    "00": ("Structure only", GREY),
    "10": ("Ionisation", BLUE),
    "01": ("Loss", ORANGE),
    "11": ("Ionisation + loss", PURPLE),
}
CDF_STYLES = {"00": "--", "10": "-", "01": (0, (7, 3)), "11": "-."}
REFERENCE = (0, (3, 3))
MODELS = {
    "GraphConv": "GraphConv",
    "Chemprop": "Chemprop",
    "ChemBERTa_head": "Frozen ChemBERTa + MLP",
}
PANEL_SIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 10.5, 9, 8, 7.5


def configure():
    rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXES,
        "axes.linewidth": .65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_heading(fig, letter, title, y):
    fig.text(HEADING_X, y, letter, fontsize=PANEL_SIZE, fontweight="bold", va="top")
    fig.text(TITLE_X, y, title, fontsize=TITLE_SIZE, fontweight="bold", va="top")
