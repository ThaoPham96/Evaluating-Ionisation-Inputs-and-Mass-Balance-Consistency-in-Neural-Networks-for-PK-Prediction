"""Assemble selected audited vector pages into an ionisation supplement."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[1]
ORIGINAL = "ionisation_results_v1"
FOLLOWUP = "ph_followup_v1"
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
INK = colors.HexColor("#202D38")
MUTED = colors.HexColor("#4B5A66")
TEAL = colors.HexColor("#176E7B")
RULE = colors.HexColor("#D3E1E5")


@dataclass(frozen=True)
class Source:
    folder: str
    pdf: str
    manifest: str
    manifest_key: str
    data: tuple[str, ...]


@dataclass(frozen=True)
class Figure:
    source: Source
    page: int
    title: str
    caption: str


OVERVIEW = Source(ORIGINAL, "ionisation_overview.pdf", "manifest.json", "outputs",
                  ("ionisation_overview_caption.txt", "ionisation_overview_plotted_data.csv"))
REPRESENTATION = Source(f"{ORIGINAL}/support", "representation_checks.pdf", "manifest.json", "outputs_sha256",
                        ("captions.md", "representation_checks_values.csv"))
DIAGNOSTICS = Source(ORIGINAL, "ionisation_diagnostics.pdf", "manifest.json", "outputs",
                     ("ionisation_diagnostics_caption.txt", "ionisation_diagnostics_plotted_data.csv"))
CONTROLS = Source(FOLLOWUP, "reference_controls.pdf", "figure_manifest.json", "output_sha256",
                  ("reference_controls.txt", "reference_controls_plotted.csv"))
INDIVIDUAL = Source(FOLLOWUP, "individual_ph_results.pdf", "figure_manifest.json", "output_sha256",
                    ("individual_ph_results.txt", "individual_ph_results_plotted.csv"))
RESULTS = Source(FOLLOWUP, "ph_main_results.pdf", "figure_manifest.json", "output_sha256",
                 ("ph_main_results.txt", "ph_main_results_plotted.csv"))
ATLAS = Source(f"{FOLLOWUP}/support", "metric_atlas.pdf", "manifest.json", "outputs_sha256",
               ("captions.md", "paired_seed_values.csv", "paired_summary.csv"))
SHIFTS = Source(f"{FOLLOWUP}/prediction_shifts", "prediction_shifts.pdf", "manifest.json", "outputs",
                ("interpretation.txt", "structure_shifts.csv", "shift_summary.csv", "seed_shifts.csv"))
BASELINE = Source("ionisation_observed_predicted_v1/baseline_comparison", "baseline_vs_ionisation.pdf",
                  "manifest.json", "outputs", ("caption.txt", "metrics.csv", "plotted_points.csv"))
ALL_ENDPOINTS = Source("ionisation_main_v2/supplementary/ph_vs_baseline_overview", "ph_vs_baseline_overview.pdf",
                       "manifest.json", "outputs", ("caption.txt", "plotted_data.csv"))


OBSERVED_PREDICTED = (
    "The top row shows structure-only models (REF0). The bottom row adds the four-channel one-hot pH 7.4 "
    "ionisation category (REF74). Columns show GraphConv, Chemprop and frozen ChemBERTa with a trained MLP head. "
    "Each point represents one structure, with its mean of nine held-out log10 predictions on the horizontal axis "
    "and observed log10 endpoint on the vertical axis. The predicted coordinate corresponds to a geometric mean in "
    "physical units. The displayed R² uses these mean predictions. The main comparison averages metrics computed "
    "separately for the nine seeds. The dashed line marks equality, and all panels share the endpoint's axes. "
    "Colours annotate the pH 7.4 threshold category in both rows. REF0 received structure input alone. "
    "Both arms use the same grouped folds, nine training seeds, equal endpoint weights and zero mass-balance penalty."
)


PAGES = (
    Figure(OVERVIEW, 1, "Category and continuous inputs for Vd prediction",
           "A and C use the audited class cohort (494 structures, 388 with observed Vd). B and D use its numerical-pKa subset (327 structures, 259 with observed Vd). A compares the one-hot pH 7.4 category with structure only. B compares acidic and basic propensities at pH 2.0, 5.0, 6.5 and 7.4 with the pH 7.4 pair alone. D also includes the repeated pH 7.4 pair, which matches the eight channels in the four-pH input. Both eight-channel inputs are divided by two after training-only standardisation. A and B show paired relative MAE changes with 95% and 99.1667% crossed-bootstrap intervals. The wider interval adjusts for the six prespecified Vd comparisons in A and B. The 494-cohort GraphConv category result excludes zero at 95% and crosses zero at 99.1667%. Chemprop and frozen ChemBERTa exclude zero at both levels for that category contrast. The numerical four-pH contrasts cross zero at the adjusted level. C and D show absolute log10 Vd MAE with seed dots, mean diamonds and sample seed SDs. The main figure reports the numerical subset's shared-recipe category versus structure-only contrast (P1−P0) using exploratory 95% intervals."),
    Figure(REPRESENTATION, 1, "Representation controls and separate tuning",
           "A compares a charge-sign scalar (anion −1, cation +1, neutral or zwitterionic 0), a four-class one-hot vector and a learned 16-dimensional embedding in the class cohort. B compares the pH 7.4 propensity pair, the pH 7.4 category, raw pKas, the four-pH propensity profile and a repeated pH 7.4 pair in the numerical subset. C and D show tuning sensitivity for one-hot versus structure only and the four-pH propensity profile versus the pH 7.4 propensity pair. Tuning uses the same cohort and folds. The class sensitivity reuses the structure-only comparator. Points show mean paired changes in log10 Vd MAE. Bars show exploratory 95% crossed-bootstrap intervals."),
    Figure(DIAGNOSTICS, 1, "Cohort, baseline and prediction diagnostics",
           "A shows the audited class counts in both cohorts. B compares Vd MAE on the 388 observed class-cohort labels with overall and class medians fitted in each training fold and Morgan-fingerprint ridge with and without class input. Deterministic controls have one trajectory. For neural models, dots show seeds, diamonds show means and bars show seed SD. C shows within-class effects of adding the pH 7.4 category (79 anionic, 139 cationic, 123 neutral and 47 zwitterionic Vd labels). Intervals are exploratory 95% crossed bootstrap. D uses predicted half-life, Vd and clearance on all 494 structures and displays mean absolute log10[t_half/(ln(2) × Vd/CL)] by seed, with mean and sample SD. The residual measures algebraic agreement among predictions. The mass-balance penalty was off."),
    Figure(CONTROLS, 1, "Categorical input and parameterisation controls",
           "All seven endpoints in the numerical subset. A compares the pH 7.4 category with structure only. B compares four repeats of that category with one copy. C compares the four-pH category representation with one pH 7.4 category block. Repeating a category changes the input parameterisation while preserving its information. All plotted 95% crossed-bootstrap intervals are exploratory."),
    Figure(INDIVIDUAL, 1, "Adding one pH at a time",
           "Each two-pH category representation (pH 7.4 plus pH 2.0, 6.5 or 5.0) is compared with two repeated pH 7.4 category blocks. Each eight-channel vector is divided by sqrt(2). Changes are shown for all seven endpoints with exploratory 95% crossed-bootstrap intervals. The pH 6.5 comparison is descriptive. The 38 category switchers at pH 6.5 include 30 with observed Vd and 23 with observed F."),
    Figure(RESULTS, 1, "Width-matched four-pH accuracy and tuning sensitivity",
           "Top panels show the categorical follow-up's six prespecified primary tests: four-pH categories (ALL4) versus four pH 7.4 copies (REP4) on Vd and F across three backbones. Bars show adjusted 99.1667% crossed-bootstrap intervals. Five intervals span zero. Frozen ChemBERTa Vd MAE increased by 2.52%, with an adjusted 99.17% interval from +0.80% to +4.27%. Lower panels show all seven endpoints under the shared recipe and separate tuning, with exploratory 95% crossed-bootstrap intervals. Both arms have 16 input channels and the same fixed scaling. Separate tuning uses the same cohort and folds and reuses some exact fitted predictions. For frozen ChemBERTa, 36 of 45 outer fits per arm were reused."),
    Figure(ATLAS, 6, "Paired changes in average absolute fold error",
           "ALL4 minus REP4 average absolute fold error (AAFE) for each endpoint under the shared recipe and separate tuning. Lower AAFE favours ALL4. Each seed difference is divided by the fixed nine-seed mean REP4 AAFE for the same model, endpoint and tuning procedure. AAFE includes only positive observed labels, including 284 of 292 observed PPB values. Dots show paired seed differences, diamonds show means and bars show ±1 sample seed SD."),
    Figure(ATLAS, 7, "Paired changes in Spearman rank correlation",
           "Raw ALL4 minus REP4 differences in Spearman rank correlation for each endpoint under the shared recipe and separate tuning. Positive values favour ALL4. Dots show paired seed differences, diamonds show means and bars show ±1 sample seed SD."),
    Figure(ATLAS, 8, "Paired changes in coefficient of determination",
           "Raw ALL4 minus REP4 differences in R² for each endpoint under the shared recipe and separate tuning. Positive values favour ALL4. R² is 1 minus the residual sum of squares divided by the total sum of squares. Dots show paired seed differences, diamonds show means and bars show ±1 sample seed SD."),
    Figure(SHIFTS, 1, "Prediction shifts across held-out structures",
           "Shared-recipe ALL4 versus REP4 predictions on all 327 structures, including structures without the corresponding observed endpoint. For each structure, paired prediction shifts are averaged across nine seeds. Positive endpoints use log2 fold ratios and F and PPB use percentage points. Each faint dot represents one structure. The circle is the median, the thick line spans the middle 50%, and the thin line spans the middle 80%. Triangles report values beyond the displayed range. These are changes in fitted predictions."),
    Figure(SHIFTS, 2, "Direction of prediction shifts by subgroup",
           "Mean signed shared-recipe ALL4 minus REP4 prediction shifts by audited pH 7.4 category and by whether the category changes at any tested pH. All 90 base-only structures and all 121 stable structures receive identical ALL4 and REP4 vectors within structure. Their predictions can differ because the models were fitted separately across the cohort. Positive endpoints use signed log2 fold ratios and F and PPB use percentage points. The shifts describe model outputs on the original PK endpoints."),
    Figure(BASELINE, 1, "Observed and predicted volume of distribution",
           "Observed and predicted Vd in the 327-structure numerical-pKa subset, with 259 labelled structures. "
           + OBSERVED_PREDICTED),
    Figure(BASELINE, 2, "Observed and predicted clearance",
           "Observed and predicted clearance in the 327-structure numerical-pKa subset, with 246 labelled structures. "
           + OBSERVED_PREDICTED),
    Figure(ALL_ENDPOINTS, 1, "MAE changes across all seven endpoints",
           "Each cell compares a categorical input arm with the same model’s structure-only reference (REF0) in the 327-structure numerical-pKa subset. Relative MAE changes use the fixed nine-seed mean REF0 MAE for that backbone and endpoint. Negative values indicate lower error. Grey labels mark repeats of the pH 7.4 category, which change input parameterisation while preserving category information. The pH 6.5 comparison is descriptive. The source table includes exploratory 95% crossed-bootstrap intervals. Figure S6 compares the four-pH categories with their width-matched repeated control."),
)


GENERAL_CAPTION = (
    "Ionisation representations were compared in a 494-structure audited class cohort and its 327-structure numerical-pKa subset. "
    "The categorical-pH follow-up used the same numerical subset and 238 frozen parent/scaffold groups. "
    "All neural metrics pool five held-out outer folds within each of nine seeds. Training used zero mass-balance penalty "
    "and seven equal endpoint weights. Separate tuning uses the same cohort and folds, with exact fitted predictions reused where applicable. "
    "Relative MAE changes and their bootstrap bounds use a fixed nine-seed mean comparator MAE for the same model, endpoint and comparison. "
    "MAE uses log10 units for half-life, Vd, CL, Cmax and Tmax, and percentage points for bioavailability (F) and plasma protein binding (PPB). "
    "Crossed intervals resample whole seed trajectories and frozen groups (10,000 draws), conditional on observed data, folds and selected settings. "
    "Seed SD describes variation across paired fits. The added pHs define representations of archived predicted pKas. PK outcomes were not measured at those pHs."
)


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def check_source(source):
    folder = FIGURES / source.folder
    manifest = folder / source.manifest
    hashes = json.loads(manifest.read_text(encoding="utf-8"))[source.manifest_key]
    result = {manifest.relative_to(FIGURES).as_posix(): digest(manifest)}
    for name in (source.pdf, *source.data):
        path = folder / name
        checksum = digest(path)
        expected = hashes.get(name)
        if expected is None and path.suffix != ".txt":
            raise ValueError(f"Missing expected source hash: {path}")
        if expected is not None and expected != checksum:
            raise ValueError(f"Source differs from its audited manifest: {path}")
        result[path.relative_to(FIGURES).as_posix()] = checksum
    return result


def paragraph(sheet, text, x, top, width, style):
    block = Paragraph(text, style)
    _, height = block.wrap(width, 1000)
    block.drawOn(sheet, x, top-height)
    return top-height


def register_fonts():
    pdfmetrics.registerFont(TTFont("Arial-Regular", FONTS / "arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", FONTS / "arialbd.ttf"))


def guide():
    stream = BytesIO()
    width, height = 760, 720
    sheet = canvas.Canvas(stream, pagesize=(width, height), invariant=True)
    sheet.setFillColor(TEAL)
    sheet.rect(0, height-6, width, 6, fill=1, stroke=0)
    sheet.setFillColor(INK)
    sheet.setFont("Arial-Bold", 29)
    sheet.drawString(44, 653, "Supplementary ionisation figures")

    body = ParagraphStyle("Body", fontName="Arial-Regular", fontSize=11.1, leading=15.2, textColor=INK, alignment=TA_LEFT)
    first = paragraph(sheet, "Ionisation representations were compared in an audited class cohort of 494 structures and its 327-structure numerical-pKa subset. The main figure uses the numerical subset for the pH 7.4 category comparison and the additional pH categories.", 44, 614, 672, body)
    second = paragraph(sheet, "All models use five grouped outer folds, nine final-training seeds and zero mass-balance penalty. Separate tuning uses the same cohort and folds and reuses some fitted predictions. In the 494-structure class cohort, the GraphConv Vd category contrast excludes zero at 95% and spans zero at the adjusted 99.1667% level.", 44, first-11, 672, body)

    top = second-29
    groups = {
        0: "Ionisation representations",
        3: "Additional pH categories",
        9: "Changes in fitted predictions",
        11: "Observed and predicted endpoints",
        13: "Accuracy across endpoints",
    }
    for i, figure in enumerate(PAGES):
        if i in groups:
            sheet.setStrokeColor(RULE)
            sheet.line(44, top+9, 716, top+9)
            sheet.setFillColor(TEAL)
            sheet.setFont("Arial-Bold", 12)
            sheet.drawString(44, top-6, groups[i])
            top -= 24
        sheet.setFillColor(INK)
        sheet.setFont("Arial-Bold", 11)
        sheet.drawString(44, top, f"S{i+1}")
        sheet.setFont("Arial-Regular", 11)
        sheet.drawString(76, top, figure.title)
        sheet.setFillColor(MUTED)
        sheet.drawRightString(716, top, str(i+2))
        top -= 23 if i+1 in groups else 19

    sheet.setFillColor(MUTED)
    sheet.setFont("Arial-Regular", 10)
    sheet.drawString(44, 52, "Dots, seed SDs and bootstrap intervals follow the legend on each source figure.")
    if top < 75:
        raise ValueError("Guide table extends into its footer")
    sheet.save()
    return PdfReader(stream).pages[0]


def add_numbered_figure(writer, page, number):
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if page.rotation != 0 or list(page.cropbox) != list(page.mediabox) or tuple(page.mediabox.lower_left) != (0, 0):
        raise ValueError("Expected an unrotated source page with matching page and crop boundaries")
    top_margin = 30
    content = writer.add_blank_page(width=width, height=height+top_margin)
    content.merge_transformed_page(page, Transformation(), over=True)
    stream = BytesIO()
    badge = canvas.Canvas(stream, pagesize=(width, height+top_margin), invariant=True)
    badge.setFillColor(TEAL)
    badge.setFont("Arial-Bold", 13)
    badge.drawString(20, height+11, f"Figure S{number}")
    badge.save()
    content.merge_page(PdfReader(stream).pages[0])
    return content


def main():
    register_fonts()
    inputs, readers = {}, {}
    for source in dict.fromkeys(figure.source for figure in PAGES):
        inputs.update(check_source(source))
        readers[source] = PdfReader(FIGURES / source.folder / source.pdf)

    writer = PdfWriter()
    writer.add_metadata({"/Title": "Supplementary ionisation figures", "/Subject": "Curated original ionisation and categorical pH results"})
    writer.add_page(guide())
    writer.add_outline_item("Reading guide", 0)
    page_map = [{"pdf_page": 1, "title": "Reading guide", "source": None}]
    captions = ["SUPPLEMENTARY IONISATION FIGURES", "", GENERAL_CAPTION]
    for number, figure in enumerate(PAGES, start=1):
        source = figure.source
        source_path = f"{source.folder}/{source.pdf}"
        if figure.page > len(readers[source].pages):
            raise ValueError(f"Missing source page {figure.page} in {source_path}")
        add_numbered_figure(writer, readers[source].pages[figure.page-1], number)
        label = f"Figure S{number}"
        writer.add_outline_item(f"{label} · {figure.title}", number)
        captions.extend(("", f"{label}. {figure.title}", figure.caption))
        page_map.append({
            "pdf_page": number+1, "figure": f"S{number}", "title": figure.title,
            "source": source_path, "source_page": figure.page, "source_sha256": inputs[source_path],
            "source_manifest": f"{source.folder}/{source.manifest}",
            "source_data_and_caption": [f"{source.folder}/{name}" for name in source.data],
        })

    pdf = HERE / "ionisation_supplementary.pdf"
    writer.write(pdf)
    caption_file = HERE / "captions.txt"
    caption_file.write_text("\n".join(captions)+"\n", encoding="utf-8")
    rendered = PdfReader(pdf)
    if len(rendered.pages) != len(PAGES) + 1 or len(rendered.outline) != len(PAGES) + 1:
        raise ValueError(f"Expected a guide and {len(PAGES)} bookmarked figures")
    for i, bookmark in enumerate(rendered.outline):
        if rendered.get_destination_page_number(bookmark) != i:
            raise ValueError("Bookmark points to a different page")
    manifest = {
        "source_paths_relative_to": "Ionisation_experiment_20261113/figures",
        "page_map": page_map, "source_sha256": inputs,
        "vector_sources": True, "source_page_transform": "Original page retained at 100% scale with a 30-point header margin",
        "builder_sha256": digest(Path(__file__)),
        "outputs_sha256": {path.name: digest(path) for path in (pdf, caption_file)},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Assembled {len(rendered.pages)} pages from {len(readers)} verified source PDFs.")


if __name__ == "__main__":
    main()
