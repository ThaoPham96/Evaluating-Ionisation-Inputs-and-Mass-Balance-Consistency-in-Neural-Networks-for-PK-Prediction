"""Visual encodings used in the coordinated ionisation figures."""

INK = "#263541"
MUTED = "#586874"
GRID = "#E5EBEF"
AXES = "#647581"
NEUTRAL_BOX = "#F1F3F4"
ROW = "#EFF3F5"

BLUE = "#0072B2"
BLUE_FILL = "#E6F1F7"
BASELINE = "#7C8993"
CONTROL = "#475763"
DIVERGING = ("#176E7B", "#F4F6F5", "#A6416B")
FONT = ["Arial", "DejaVu Sans"]

MODELS = {
    "GraphConv": "GraphConv",
    "Chemprop": "Chemprop",
    "ChemBERTa_head": "Frozen ChemBERTa + MLP",
}
ENDPOINT_COLORS = {
    "t_half": "#B2182B",
    "Vd": "#00865A",
    "CL": "#9A7B00",
    "Cmax": "#C04B8B",
    "Tmax": "#596B2F",
    "F": "#238C8C",
    "PPB": "#6B4530",
}

CATEGORY_EDGE = "#333333"
CATEGORIES = {
    "anionic": ("Acid-only", "#24246B", "o"),
    "cationic": ("Base-only", "#00BFC4", "^"),
    "neutral": ("Neither-active", "#EFCB00", "s"),
    "zwitterionic": ("Dual-active", "#FF796C", "D"),
}
AUDITED_CLASSES = ("Anionic", "Cationic", "Neutral", "Zwitterionic")
