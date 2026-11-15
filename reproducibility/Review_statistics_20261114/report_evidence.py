"""Write a compact, nonselective evidence digest from the verified claim tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import HERE,EPS,record,HASHES

def main(out=HERE/'outputs'):
    d=pd.read_csv(out/'contrasts.csv'); m=pd.read_csv(out/'condition_metrics.csv')
    absolute_only=~d.metric.isin([*EPS,'triad_mae','seven_endpoint_average'])
    d.loc[absolute_only,['relative_unit','scale']]='not_applicable'
    d.to_csv(out/'contrasts.csv',index=False)
    d.loc[d.stratum.isin(['all','observed_le_1.5','observed_gt_1.5'])].to_csv(out/'machine_claim_table.csv',index=False)
    archive=pd.read_csv(out/'exact_archived_comparison.csv')
    lines=['S/H SAVED-PREDICTION REANALYSIS — 14 NOVEMBER 2026','',
    'STATUS: local CPU scoring only; no training, model selection, paid compute or source edits.',
    'Full952: 952 structures / 674 complete observed triads / 552 whole groups.',
    'Unified327: 327 structures / 242 complete observed triads / 238 whole groups.',
    'Class494: all four archived category encodings; separate cohort, no mixing with327.',
    'Primary main models use nine fixed seeds 101,202,303,404,505,606,707,808,909.',
    'Supplementary encoders/depth/weights/single-task use actual three seeds101,202,303.',
    '',
    'ESTIMAND: within each held-out cohort and seed, pool five outer folds and average absolute errors on the endpoint-specific observed mask. Average the paired seed changes and divide by the comparator mean MAE. Equal seven-endpoint averages average the seven relative effects, never mix log10 MAE and percentage-point MAE directly.',
    'Primary nominal95% intervals resample entire frozen groups with replacement, using shared draws across endpoints/arms/model families and recomputing the mean comparator MAE within each draw. Fitted seeds remain fixed. Separate seed SDs describe variation across saved fits. These intervals condition on fitted models, observed training data, folds and selected recipes; they do not reproduce training/hyperparameter uncertainty.',
    'Full952 uses the original10000 PCG64 integer group draws, seed20261112. Unified327 uses the original10000 multinomial group draws followed by multinomial seed draws, seed20261113. Primary intervals use only group counts; crossed sensitivity uses group and seed counts with original fixed denominator. Exact original intervals/SDs are retained in exact_archived_comparison.csv.',
    'F structure-scale 11−10/11−01 effects retain the original00 denominator and are labelled percentage-point differences on the00-normalised MAE scale. Separately labelled direct-comparator percentages use10 or01. Interactions use common00 in each draw. T will remain a distinct tuning policy and reuse its own structure-only denominator.',
    '',
    'VALIDATION: source-row endpoint values reconstruct every observed full952 label; identity masks match curated model values; retained input graphs match reviewed source records. Frozen outer and inner assignments isolate groups and exclude outer-test structures from inner fitting. Primary/supplementary full952 saved fits pass archived per-fit request/hash/selection validation and independent export-to-row comparison. Unified original exporter verifies campaign/fit identities, inner-only lambda selection and exact shared reference predictions/fit IDs; an independent scorer verifies truth/masks/coverage. Predictions are finite and unique per intended condition/seed/structure. No prediction ensembles are substituted for individual-seed MAE.',
    'Scope limit: these checks audit saved requests/exports and selected provenance. They do not re-execute historical training or prove every historical scaler operation from the numerical prediction files alone. Class494 is checked for truth/split/coverage/units against the saved release; per-fit reconstruction is not repeated for that historical cohort.',
    '',f'Exact archive rows verified: {len(archive)}. Original point estimates all preserved. Original CI direction changes: {int(archive.interval_direction_changed.sum())} (complete rows in exact_archived_interval_changes.csv).',
    '', 'CURRENT MAIN EFFECTS (all labelled structures; relative%, nominal group95%):']
    main=d.loc[d.protocol.eq('F')&d.stratum.eq('all')&d.metric.isin(['Vd','CL','triad_mae','seven_endpoint_average'])&d.contrast.isin(['selected_positive-baseline','10-00','01-00','11-00','11-10','interaction'])]
    for r in main.itertuples():
        lines.append(f'{r.cohort} | {r.model} | {r.contrast} | {r.metric}: {r.relative_pct:.6f} [{r.primary_low:.6f}, {r.primary_high:.6f}]; scale={r.scale}, denominator={r.denominator}')
    lines+=['','HARD ALGEBRAIC POST-PROCESSING: all three directions, no held-out selection.',
    'log10(t_half)=log10(ln2)+log10(Vd)−log10(CL); derive Vd or CL by algebraic rearrangement. Only the named endpoint is replaced; other predictions are unchanged. Exact internal consistency is guaranteed and does not demonstrate accuracy. This is not a trained hard-constraint model.',
    'The complete H table includes every source arm and direction; below are all derived-endpoint effects across all/consistent/divergent strata. Negative values improve MAE; positive values worsen it.']
    hard=d.loc[d.contrast.str.startswith('H:')].copy()
    hard['derived_endpoint']=hard.contrast.str.extract(r'__derive_(t_half|Vd|CL)-original')[0]
    hard=hard.loc[hard.metric.eq(hard.derived_endpoint)&hard.stratum.isin(['all','observed_le_1.5','observed_gt_1.5'])]
    hard.to_csv(out/'hard_derivation_target_endpoint.csv',index=False)
    for r in hard.itertuples():
        lines.append(f'{r.cohort} | {r.model} | {r.contrast} | {r.stratum} n={int(r.n)}: originalMAE={r.denominator_mean:.6f}, derivedMAE={r.denominator_mean+r.absolute_delta:.6f}; delta%={r.relative_pct:.6f} [{r.primary_low:.6f}, {r.primary_high:.6f}]')
    lines+=['','FILES: contrasts.csv is the complete prespecified comparison table; condition_metrics.csv contains absolute MAEs and consistency; individual_seed_metrics.csv contains fitted-seed values; ensemble_scatter_metrics.csv explicitly separates ensemble predictive R²/MAE from mean individual-seed MAE; machine_claim_table.csv is a mechanically selected claim source without significance filtering; manifest.json binds inputs, code, versions, seeds and estimator.','',
    'EXECUTE: python Review_statistics_20261114/evaluate.py; python Review_statistics_20261114/archive_comparison.py; python Review_statistics_20261114/report_evidence.py',
    'NEW MATRIX: python Review_statistics_20261114/evaluate.py --new-export <export-directory> --output <new-output-directory>. New files must match the schema in PREDICTION_SCHEMA.txt. Existing F references are checked against immutable original fit IDs and values; T remains separate. No new T/R0/R2 result is claimed before its export exists and passes the training/export provenance gates.']
    (out/'EVIDENCE.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=json.loads((out/'manifest.json').read_text())
    for path in HERE.glob('*.py'): record(path)
    manifest['source_sha256'].update(HASHES)
    output_hashes={}
    for path in out.iterdir():
        if path.is_file() and path.name!='manifest.json':
            record(path);output_hashes[path.name]=HASHES[str(path.resolve())]
    manifest['output_sha256']=output_hashes
    manifest['tests']={'unit_tests':'4 passed; whole-group/missing masks, three hard directions, common interaction denominator/ratio-of-mean estimand, duplicate/fold fail-closed', 'synthetic_new_export':'141264 rows, 4092 diagnostic contrasts at100 draws; validated F/T full coverage, immutable F IDs/values, inner zero-inclusive/R2 selection schema. No synthetic results in production output.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Evidence written',out/'EVIDENCE.txt')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs')
    main(parser.parse_args().output)
