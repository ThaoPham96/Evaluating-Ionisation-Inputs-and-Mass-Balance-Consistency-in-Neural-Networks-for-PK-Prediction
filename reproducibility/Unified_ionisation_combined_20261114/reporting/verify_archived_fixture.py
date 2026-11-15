"""Exercise reporting against archival data in a disposable, non-campaign fixture."""
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import build as report


def verify() -> None:
    score = report.score
    old = report.OLD_COMBINED / "outputs/export"
    manifest, final, _, _, selections = score.read_export(old)
    keys = ["structure_id", "backbone", "outer_fold", "seed"]
    refs = final.loc[final.cell.isin(report.REF.values()), [*keys, "cell", "fit_id", *score.PRED_COLUMNS]].copy()
    refs = refs.rename(columns={"cell": "arm", "fit_id": "source_fit_id"}).assign(
        arm=lambda rows: rows.arm.map({"00": "P0", "10": "P1"}), cohort="numerical", procedure="primary",
    )
    for endpoint in score.ENDPOINTS:
        q = refs[f"q_{endpoint}"]
        if endpoint in ("F", "PPB"):
            refs[f"physical_{endpoint}"] = q
            refs[f"pred_{endpoint}"] = np.log10(q / (100 - q))
        else:
            refs[f"pred_{endpoint}"] = q
            refs[f"physical_{endpoint}"] = 10**q
    refs.drop(columns=score.PRED_COLUMNS, inplace=True)
    archived_alts = []
    for directory in ("single_ph", "ph8"):
        p = score.read_csv(report.OLD_ION / "outputs" / directory / "outer_predictions.csv")
        archived_alts.append(p.loc[p.arm.isin(report.ALT_PH)])
    ph = pd.concat([refs.assign(arm=lambda rows: rows.arm.map(report.ALIASES)), *archived_alts], ignore_index=True)
    assert len(ph) == 52974
    # Use a deliberately impossible campaign identity. Neither source archive
    # substitutes for a unified output; no files from this fixture are retained.
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        export = root / "exports"
        combo, ion = export / "combined", export / "ionisation"
        combo.mkdir(parents=True)
        ion.mkdir()
        for filename in score.EXPORT_FILES[:3]:
            copyfile(old / filename, combo / filename)
        identity = {"counts": report.FIT_COUNTS, "fixture": True}
        fake_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report.write_json(export / "plan.json", dict(campaign_id=fake_id, identity=identity))
        for row in selections:
            row["campaign_id"] = fake_id
        report.write_json(combo / "selections.json", selections)
        base = dict(campaign_id=fake_id, release_fingerprint=score.FINGERPRINT, plan_sha256=report.digest(export / "plan.json"),
                    environment_hash="fixture", unified_fit_counts=report.FIT_COUNTS)
        manifest["files"]["selections.json"] = report.digest(combo / "selections.json")
        report.write_json(combo / "manifest.json", {**manifest, **base})
        refs.assign(source_campaign_id=fake_id).to_csv(ion / "outer_predictions.csv", index=False)
        ph.assign(source_campaign_id=fake_id).to_csv(ion / "reference_and_ph_predictions.csv", index=False)
        report.write_json(ion / "manifest.json", dict(base, rows=len(ph), reference_rows=len(refs), fits=810, reference_fits=270,
                                                     files={p.name: report.digest(p) for p in ion.glob("*.csv")}))
        report.write_json(export / "manifest.json", dict(base, unit_markers={str(i): "fixture" for i in range(15)},
                         exports={p.name: report.digest(p / "manifest.json") for p in (combo, ion)}))
        dest = root / "staged"
        (dest / "scores").mkdir(parents=True)
        pinned_id = report.CAMPAIGN_ID
        try:
            report.CAMPAIGN_ID = fake_id
            report.build(export, dest)
        finally:
            report.CAMPAIGN_ID = pinned_id
        keys = ["panel", "backbone", "metric", "contrast", "plot_kind", "seed", "cell"]
        old_plot = report.OLD_COMBINED / "figures/combined_main_v2/plot_data.csv"
        a, g = [score.read_csv(path).set_index(keys).sort_index() for path in (old_plot, dest / "fig3/plot_data.csv")]
        pd.testing.assert_index_equal(a.index, g.index)
        np.testing.assert_allclose(g.to_numpy(dtype=float), a.to_numpy(dtype=float), rtol=0, atol=1e-12, equal_nan=True)
        receipt = report.read_json(dest / "manifest.json")["rendering"]
        print("Reproduced all 840 archived Figure 3 source rows and rendered three staged PDFs.")
        print("Shared Figure 2 and Figure 3:", receipt["ph74_identity"])
        print("Supplementary bounds:", receipt["figS2"]["artists_bbox_inches"])


if __name__ == "__main__":
    verify()
