"""Bundle the five generated supplementary PDFs; run with system Python."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
DEST = HERE / "supplementary"
STEMS = ["S1_interactions", "S2_posthoc_controls", "S3_accuracy_and_disagreement", "S4_paired_effects", "S5_matched_triads"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    writer = PdfWriter()
    for stem in STEMS:
        path = DEST / f"{stem}.pdf"
        assert len(PdfReader(path).pages) == 1
        writer.append(str(path), outline_item=stem.replace("_", " "))
    output = DEST / "combined_supplementary.pdf"
    writer.write(str(output))
    assert len(PdfReader(output).pages) == 5
    record = {"pages": 5, "builder_sha256": sha(Path(__file__)),
              "main_render_sha256": sha(HERE / "render.py"),
              "files": {p.name: sha(p) for p in DEST.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (DEST / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supplementary_pdf": str(output), "pages": 5}))
