"""Package the other result plots as S1-S5, preserving the source figures.

Run with system Python (pypdf available). The render subprocess uses the existing
research environment for matplotlib and pandas; no packages are installed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / 'supplementary'
PREVIOUS = HERE.parent / 'combined_supplementary_v1'
REVIEW = HERE.parent / 'combined_results_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_s4():
    import pandas as pd

    spec = importlib.util.spec_from_file_location('combined_review_plots', REVIEW/'render.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, frames = module.load_scores(ROOT/'outputs/scores')
    means, seeds, cdf, _, _ = module.extract(frames)
    original_save = module.save
    def save_s4(fig, destination, stem):
        fig.texts[0].set_text('S4  Absolute accuracy and continuous disagreement')
        original_save(fig, destination, 'S4_accuracy_and_disagreement')
    module.save = save_s4
    DEST.mkdir(parents=True, exist_ok=True)
    omitted, denominator = module.overview(means, seeds, cdf, DEST)
    pd.concat([means.reset_index().assign(panel='S4A', kind='summary'),
               seeds.reset_index().assign(panel='S4A', kind='seed'),
               cdf.assign(panel='S4B', kind='cdf')], ignore_index=True).to_csv(DEST/'S4_plot_data.csv', index=False)
    (DEST/'S4_render_manifest.json').write_text(json.dumps(dict(campaign_id=manifest['campaign_id'],
        source_render_sha256=sha(REVIEW/'render.py'), score_manifest_sha256=sha(ROOT/'outputs/scores/manifest.json'),
        cdf_view_max_fold=5, max_predictions_above_view=omitted, seed_predictions_per_curve=denominator), indent=2)+'\n')


def package():
    from pypdf import PdfReader, PdfWriter

    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT.parent/'revision_work/.venv/Scripts/python.exe'),
                    str(Path(__file__)), '--render-only'], check=True)
    mappings = [('all_endpoints','S1_all_endpoints'), ('interactions','S2_interactions'),
                ('posthoc_controls','S3_posthoc_controls')]
    source_hashes = {}
    for old, new in mappings:
        for suffix in ('pdf','png','svg'):
            source, target = PREVIOUS/f'{old}.{suffix}', DEST/f'{new}.{suffix}'
            shutil.copy2(source, target)
            assert sha(source) == sha(target)
            source_hashes[str(source.relative_to(ROOT)).replace('\\','/')] = sha(source)
    shutil.copy2(PREVIOUS/'plot_data.csv', DEST/'S1_S3_plot_data.csv')
    caption = (PREVIOUS/'captions.txt').read_text(encoding='utf-8')
    prior = (REVIEW/'caption.txt').read_text(encoding='utf-8').split('Adding the loss to ionisation-informed models')[0]
    caption += '\nS4: '+prior.replace('Accuracy and internal consistency across four conditions\n\n', '')
    caption += '\n' + (DEST/'S5_caption.txt').read_text(encoding='utf-8')
    (DEST/'captions.txt').write_text(caption, encoding='utf-8')
    writer = PdfWriter()
    labels = [(new, new.replace('_',' ')) for _,new in mappings]
    labels += [('S4_accuracy_and_disagreement', 'S4 Absolute accuracy and continuous disagreement')]
    labels += [('S5_paired_effects', 'S5 Paired accuracy and consistency effects')]
    for stem, label in labels:
        writer.append(str(DEST/f'{stem}.pdf'), outline_item=label)
    output = DEST/'combined_supplementary.pdf'
    with output.open('wb') as stream:
        writer.write(stream)
    assert len(PdfReader(output).pages) == 5
    files = {p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(dict(campaign_id=json.loads((ROOT/'outputs/scores/manifest.json').read_text())['campaign_id'],
        pages=5, builder_sha256=sha(__file__), main_render_sha256=sha(HERE/'render.py'),
        copied_source_hashes=source_hashes, files=files), indent=2)+'\n')
    print(json.dumps({'supplementary_pdf':str(output), 'pages':5}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    render_s4() if args.render_only else package()
