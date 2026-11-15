"""Additional summaries of archived structures and held-out predictions; no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits


def main(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'Review_statistics_20261114'))
    import evaluate as ev
    from sources import aligned
    ev.HASHES.clear()
    mass = root / 'Mass_balance_experiment_20261113'
    unified = root / 'Unified_ionisation_combined_20261114'
    full = ev.cohort('full952', mass / 'data', mass / 'data/same_row_complete_triads.csv')
    subset = ev.cohort('unified327', unified / 'data/release/numerical', unified / 'runtime/same_row_complete_triads.csv')
    classes = ev.read(root / 'Ionisation_experiment_20261113/data/release/class/model_dataset.csv')
    details = []
    for sid, row in full.rows.iterrows():
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        assert mol is not None, sid
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        details.append(dict(structure_id=sid, component_count=len(Chem.GetMolFrags(mol)),
                            charged_atom_count=sum(q != 0 for q in charges), net_formal_charge=sum(charges)))
    detail = pd.DataFrame(details)
    rows = []
    for name, ids in [('full952', full.rows.index), ('class494', classes.structure_id), ('numerical327', subset.rows.index)]:
        d = detail[detail.structure_id.isin(ids)]
        assert len(d) == len(ids)
        for field, mask in [('multiple_components', d.component_count.gt(1)),
                            ('any_charged_atom', d.charged_atom_count.gt(0)),
                            ('nonzero_net_formal_charge', d.net_formal_charge.ne(0))]:
            rows.append(dict(cohort=name, feature=field, n=int(mask.sum()), denominator=len(d), percent=100 * mask.mean()))
    pd.DataFrame(rows).to_csv(output / 'Supplementary_Data_5_structure_summary.csv', index=False)
    detail.to_csv(output / 'Supplementary_Data_5_structure_flags.csv', index=False)

    def contrast(c, boot, arrays, mask, model, analysis, metric, terms, denominator, unit='log10'):
        stats = {k: boot.means(v, mask) for k, v in arrays.items()}
        points = sum(w * stats[k][0] for k, w in terms.items())
        draws = sum(w * stats[k][1] for k, w in terms.items())
        bp, bd = stats[denominator]
        alo, ahi = ev.bounds(draws.mean(axis=1))
        lo, hi = ev.bounds(100 * draws.mean(axis=1) / bd.mean(axis=1))
        return dict(cohort=c.name, model=model, analysis=analysis, metric=metric, unit=unit,
                    n=int(mask.sum()), n_groups=int(np.unique(c.group[mask]).size), n_seeds=len(ev.SEEDS),
                    terms=json.dumps(terms, sort_keys=True), denominator=denominator,
                    denominator_mean=float(bp.mean()), condition_means=json.dumps({k:float(v[0].mean()) for k,v in stats.items()},sort_keys=True),
                    absolute_delta=float(points.mean()), absolute_low=alo, absolute_high=ahi,
                    relative_pct=float(100 * points.mean() / bp.mean()), primary_low=lo, primary_high=hi)

    table = ev.read(mass / 'figure_data/primary_outer_predictions.csv.gz')
    partial = np.isfinite(full.truth[:, :3]).sum(axis=1)
    assert dict(zip(*np.unique(partial, return_counts=True))) == {0:45, 1:187, 2:46, 3:674}
    partial_mask = (partial >= 1) & (partial <= 2)
    boot = ev.Bootstrap(full, ev.SEEDS)
    results = []
    for model in ['GraphConv', 'Chemprop', 'ChemBERTa_head']:
        arms = {}
        for policy in ['baseline', 'selected_positive']:
            d = table[table.backbone.eq(model) & table.policy.eq(policy)].copy()
            for ep in ev.EPS:
                suffix = ep + ('_percent' if ep in ['F', 'PPB'] else '_log10')
                d['q_' + ep] = d['pred_' + suffix]
                d['true_' + ep] = d['observed_endpoint_' + suffix]
                d['observed_' + ep] = d['true_' + ep].notna()
            arms[policy] = aligned(d, full, ev.SEEDS)
        for j, ep in enumerate(ev.EPS[:3]):
            mask = partial_mask & np.isfinite(full.truth[:, j])
            errors = {k:np.abs(q[:, :, j] - full.truth[:, j]) for k,q in arms.items()}
            results.append(contrast(full, boot, errors, mask, model, 'one_or_two_observed_triad_endpoints', ep,
                                    {'selected_positive':1, 'baseline':-1}, 'baseline'))
    pd.DataFrame(results).to_csv(output / 'Supplementary_Data_5_partial_triad_accuracy.csv', index=False)

    table = ev.read(root / 'Review_robustness_experiment_20261114/outputs/export/final_predictions.csv')
    boot = ev.Bootstrap(subset, ev.SEEDS)
    reference, interactions = [], []
    for model in ['GraphConv', 'Chemprop', 'ChemBERTa_head']:
        d = table[table.backbone.eq(model)]
        arrays = {}
        for protocol, inp, penalty in [('F','structure','off'),('F','structure','shared'),('F','structure','separate'),
                                       ('F','category','off'),('F','category','shared'),('F','category','separate'),
                                       ('T','structure','off'),('T','structure','positive')]:
            arrays[f'{protocol}_{inp}_{penalty}'] = aligned(d[d.protocol.eq(protocol) & d.input.eq(inp) & d.penalty.eq(penalty)], subset, ev.SEEDS)
        metrics = {name:ev.metrics(q, subset) for name,q in arrays.items()}
        for fpolicy, tpolicy in [('off','off'),('separate','positive'),('shared','positive')]:
            fkey, tkey = f'F_structure_{fpolicy}', f'T_structure_{tpolicy}'
            for metric in ['t_half','Vd','CL','triad_mae','mean_abs_residual']:
                key = ('complete_triad' if metric == 'triad_mae' else 'all', metric)
                values = {k:metrics[k][key][0] for k in [fkey, tkey]}
                reference.append(contrast(subset,boot,values,metrics[fkey][key][1],model,
                    f'T_{tpolicy}_minus_F_{fpolicy}',metric,{tkey:1,fkey:-1},fkey))
        for policy in ['shared','separate']:
            terms = {f'F_category_{policy}':1,'F_category_off':-1,f'F_structure_{policy}':-1,'F_structure_off':1}
            for metric in list(ev.EPS) + ['triad_mae','mean_abs_residual']:
                key = ('complete_triad' if metric == 'triad_mae' else 'all', metric)
                values = {k:metrics[k][key][0] for k in terms}
                interactions.append(contrast(subset,boot,values,metrics['F_structure_off'][key][1],model,
                    f'F_{policy}_interaction',metric,terms,'F_structure_off',metrics['F_structure_off'][key][2]))
    pd.DataFrame(reference).to_csv(output / 'Supplementary_Data_5_F_T_reference_comparison.csv', index=False)
    pd.DataFrame(interactions).to_csv(output / 'Supplementary_Data_5_F_penalty_interactions.csv', index=False)
    manifest = dict(bootstrap_draws=10000, bootstrap_seeds={'full952':20261112,'unified327':20261113},
                    estimator='Nine seed means; paired whole-group percentile intervals with fitted models fixed and denominator recomputed per draw.',
                    partial_triad_structures=int(partial_mask.sum()),
                    structures='Disconnected components include hydrates/formulation components; nonzero atomic formal charges also include charge-separated valence representations.',
                    reference_scope='F shared versus T positive changes penalty policy and calibration; F separate versus T positive uses the same per-input selection rule.',
                    input_sha256={str(Path(p).relative_to(root)):h for p,h in ev.HASHES.items()},
                    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('Supplementary_Data_5_*.csv')})
    (output / 'Supplementary_Data_5_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(structure_counts=rows,partial_accuracy=results,reference_rows=len(reference),interaction_rows=len(interactions)),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.root.resolve(),args.output)
