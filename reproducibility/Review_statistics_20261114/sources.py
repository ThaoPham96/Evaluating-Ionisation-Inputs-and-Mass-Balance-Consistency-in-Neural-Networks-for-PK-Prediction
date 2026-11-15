"""Read-only adapters, identity/split validation and prespecified contrasts."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from evaluate import ROOT, EPS, SEEDS, HASHES, cohort, read, record, require

MASS=ROOT/'Mass_balance_experiment_20261113'
UNI=ROOT/'Unified_ionisation_combined_20261114'

def pair(a,b): return (f'{a}-{b}',{a:1,b:-1},b)

def aligned(d,c,seeds,prefix='q_'):
    d=d.set_index(['seed','structure_id'],verify_integrity=True).sort_index()
    ix=pd.MultiIndex.from_product([seeds,c.rows.index],names=['seed','structure_id'])
    require(d.index.equals(ix),'Incomplete or extra seed/structure rows')
    ids=d.index.get_level_values('structure_id')
    require(np.array_equal(d.outer_fold,ids.map(c.rows.outer_fold)),'Held-out fold mismatch')
    if 'group_id' in d: require(np.array_equal(d.group_id,ids.map(c.rows.group_id)),'Prediction group mismatch')
    if 'same_row_triad_eligible' in d:
        require(np.array_equal(d.same_row_triad_eligible.to_numpy(bool),np.tile(c.triads,len(seeds))),'Stored triad mask mismatch')
    if 'true_t_half' in d:
        truth=np.tile(c.truth,(len(seeds),1))
        np.testing.assert_allclose(d[[f'true_{e}' for e in EPS]],truth,rtol=0,atol=1e-14,equal_nan=True)
        np.testing.assert_array_equal(d[[f'observed_{e}' for e in EPS]].to_numpy(bool),np.isfinite(truth))
    cols=[f'{prefix}{e}' for e in EPS] if prefix else [('physical_' if j>=5 else 'pred_')+e for j,e in enumerate(EPS)]
    q=d[cols].to_numpy().reshape(len(seeds),len(c.rows),7)
    require(np.isfinite(q).all(),'Nonfinite held-out predictions')
    require(((q[:,:,5:]>=0)&(q[:,:,5:]<=100)).all(),'Physical percentage predictions outside0–100')
    return q

def existing(ev,skip_supp=False):
    sys.dont_write_bytecode=True
    # Import the archived validators without running their reporting entrypoints.
    sys.path.insert(0,str(MASS/'supplementary_completion/code'))
    import mbsupp as mb
    mass=cohort('full952',MASS/'data',MASS/'data/same_row_complete_triads.csv')
    require((len(mass.rows),mass.triads.sum(),mass.rows.group_id.nunique())==(952,674,552),'Full cohort count mismatch')
    source=mb.Source()
    source_row_audit(mass)
    order=[source.cohort.index[s] for s in mass.rows.index]
    eporder=[mb.NAMES.index(e) for e in EPS]
    def conv(raw,seeds): return mb.units(raw,seeds)[:,order][:,:,eporder]
    table=read(MASS/'figure_data/primary_outer_predictions.csv.gz')
    for model in mb.PRIMARY:
        arms={}
        for policy in ('baseline','selected_positive','zero_eligible'):
            q=conv(source.primary(model,policy,SEEDS),SEEDS)
            d=table.loc[table.backbone.eq(model)&table.policy.eq(policy)].copy()
            for ep in EPS:
                suffix=ep+('_percent' if ep in ('F','PPB') else '_log10')
                d['q_'+ep]=d['pred_'+suffix]
                d['true_'+ep]=d['observed_endpoint_'+suffix]
                d['observed_'+ep]=d['true_'+ep].notna()
            np.testing.assert_allclose(q,aligned(d,mass,SEEDS),atol=2e-13,rtol=2e-13)
            arms[policy]=q
        ev.suite(mass,model,'F',arms,SEEDS,[pair('selected_positive','baseline'),pair('zero_eligible','baseline'),pair('zero_eligible','selected_positive')],hard=True,legacy='group_recomputed')
        print('Verified/scored full',model,flush=True)
    if not skip_supp:
        seeds=SEEDS[:3]
        ref={k:conv(source.primary('GraphConv',p,seeds),seeds) for k,p in [('baseline','baseline'),('loss','selected_positive')]}
        ev.suite(mass,'GraphConv_depth2','F_supp3',ref,seeds,[pair('loss','baseline')],legacy='group_recomputed')
        for block,cells in mb.SECONDARY_BLOCKS.items():
            for model,depth in cells:
                arms={arm:conv(source.secondary(block,model,depth,arm,seeds),seeds) for arm in ('baseline','loss')}
                ev.suite(mass,f'{model}_depth{depth}_{block}','F_supp3',arms,seeds,[pair('loss','baseline')],legacy='group_recomputed')
        single=conv(source.single_task(seeds),seeds)
        ev.suite(mass,'GraphConv_single_vs_multi','F_supp3',dict(single_task=single,**ref),seeds,[pair('single_task','baseline'),pair('single_task','loss')],legacy='group_recomputed')
        print('Verified/scored supplementary encoders/depth/weights/single-task',flush=True)
    HASHES.update({str(source.path/p):v for p,v in source.source_hashes.items()})
    ev.audit.append(dict(cohort='full952',structures=952,complete_triads=674,groups=552,validator='Archived mbsupp.Source validates per-fit request/content hashes, expected train/predict IDs, selected config/lambda/epochs; predictions independently joined to truth/group/fold/masks and frozen export.',verified_source_files=len(source.source_hashes)))

    # Unified original exporter/selection validator, plus independent array scorer.
    sys.path.insert(0,str(UNI/'reporting')); sys.path.insert(0,str(UNI/'analysis'))
    spec=importlib.util.spec_from_file_location('original_unified_reporting',UNI/'reporting/build.py')
    build=importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    oldc=build.score.load_cohort(UNI/'data/release/numerical',UNI/'runtime/same_row_complete_triads.csv')
    _,validated,_,_,selections,_=build.verify_exports(UNI/'outputs/export',oldc)
    unified=cohort('unified327',UNI/'data/release/numerical',UNI/'runtime/same_row_complete_triads.csv')
    require((len(unified.rows),unified.triads.sum(),unified.rows.group_id.nunique())==(327,242,238),'Unified cohort count mismatch')
    final=read(UNI/'outputs/export/combined/final_predictions.csv')
    ph=read(UNI/'outputs/export/ionisation/reference_and_ph_predictions.csv')
    for model in build.score.BACKBONES:
        arms={cell:aligned(final.loc[final.backbone.eq(model)&final.cell.eq(cell)],unified,SEEDS) for cell in ('00','10','01','11')}
        selected=build.score.zero_eligible_predictions(validated[model],oldc,selections,model)
        for key in arms: np.testing.assert_array_equal(arms[key],selected[key])
        comps=[pair('10','00'),pair('01','00'),pair('11','00'),pair('11','10'),pair('11','01'),('11-10_structure_scale',{'11':1,'10':-1},'00'),('11-01_structure_scale',{'11':1,'01':-1},'00'),('interaction',{'11':1,'10':-1,'01':-1,'00':1},'00')]
        ev.suite(unified,model,'F',arms,SEEDS,comps,hard=True,legacy='crossed_fixed')
        sensitivity={key:selected[key] for key in ('00','10','zero_eligible_01','zero_eligible_11')}
        ev.suite(unified,model,'F_zero_policy',sensitivity,SEEDS,[pair('zero_eligible_01','00'),('zero_eligible_11-10_structure_scale',{'zero_eligible_11':1,'10':-1},'00'),('zero_eligible_interaction',{'zero_eligible_11':1,'10':-1,'zero_eligible_01':-1,'00':1},'00')],legacy='crossed_fixed')
        pharms={'REF0':arms['00'],'REF74':arms['10']}
        pharms.update({arm:aligned(ph.loc[ph.backbone.eq(model)&ph.arm.eq(arm)],unified,SEEDS,prefix='') for arm in ('PH2','PH5','PH65','PH8')})
        ev.suite(unified,model,'F_pH',pharms,SEEDS,[pair(a,b) for a in ('PH2','PH5','PH65','REF74','PH8') for b in ('REF0','REF74') if a!=b],legacy='crossed_fixed')
        print('Verified/scored unified',model,flush=True)
    for path in (UNI/'outputs/export').rglob('*'):
        if path.is_file(): record(path)
    ev.audit.append(dict(cohort='unified327',structures=327,complete_triads=242,groups=238,validator='Original unified verify_exports validates campaign and source hashes, all fit identities, inner-only lambda selection and exact reference predictions/fit IDs across Fig2/Fig3 exports; independent truth/fold/group/mask checks.'))
    if not skip_supp: class_encoding(ev)

def source_row_audit(c):
    from evaluate import RAW
    provenance=read(MASS/'data/endpoint_provenance.csv')
    mapping=read(MASS/'data/source_row_mapping.csv')
    reviewed=read(MASS/'data/reviewed_source_records.csv').set_index('ORIG_INDEX',verify_integrity=True)
    masks=read(MASS/'data/endpoint_identity_masks.csv')
    links=set(zip(mapping.row_idx,mapping.structure_id))
    observed=set()
    for r in provenance.itertuples(index=False):
        require((r.row_idx,r.structure_id) in links,'Endpoint provenance source mapping mismatch')
        require(r.structure_id in c.rows.index and r.endpoint in RAW,'Unexpected endpoint provenance')
        require(float(c.rows.loc[r.structure_id,r.endpoint])==float(r.value),'Endpoint value differs from source provenance')
        require(float(reviewed.loc[r.row_idx,r.endpoint])==float(r.value),'Endpoint differs from reviewed source row')
        observed.add((r.structure_id,r.endpoint))
    expected={(sid,ep) for sid,row in c.rows.iterrows() for ep in RAW if pd.notna(row[ep])}
    require(observed==expected,'Observed targets lack complete source provenance')
    for r in masks.itertuples(index=False):
        ids=mapping.loc[mapping.row_idx.eq(r.row_idx),'structure_id']
        for sid in ids:
            if sid in c.rows.index:
                expected_value=r.model_value
                actual=c.rows.loc[sid,r.endpoint]
                require((pd.isna(actual) and pd.isna(expected_value)) or actual==expected_value,'Masked identity endpoint differs from curated model value')
    # Input graphs are preserved exactly from reviewed records; neutralisation is not applied.
    for sid,row in c.rows.iterrows():
        source_ids=json.loads(row.source_rows)
        require(all((s,sid) in links for s in source_ids),'Dataset source_rows mapping mismatch')
        require(any(reviewed.loc[s,'canonical_smiles']==row.canonical_smiles for s in source_ids),'Retained input graph not in reviewed source records')

def class_encoding(ev):
    old=ROOT/'Ionisation_experiment_20261113'
    # Same-row triads are intersected explicitly, preserving provenance.
    fulltriad=read(MASS/'data/same_row_complete_triads.csv')
    data=old/'data/release/class'
    rows=read(data/'model_dataset.csv').set_index('structure_id',verify_integrity=True).sort_index()
    from evaluate import Cohort,RAW
    outer=read(data/'outer.csv').set_index('structure_id',verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(rows[['group_id','outer_fold']],outer[['group_id','outer_fold']],check_dtype=False)
    require(rows.groupby('group_id').outer_fold.nunique().max()==1,'Class groups cross folds')
    inner=read(data/'inner.csv')
    for fold,subset in inner.groupby('outer_fold'):
        require(len(subset)==len(rows.loc[rows.outer_fold.ne(fold)]) and set(subset.structure_id)==set(rows.index[rows.outer_fold.ne(fold)]),'Class inner/outer leakage or missing IDs')
        require(subset.groupby('group_id').inner_fold.nunique().max()==1,'Class inner group leakage')
    full=read(MASS/'data/model_dataset.csv').set_index('structure_id',verify_integrity=True)
    np.testing.assert_allclose(rows[list(RAW)],full.loc[rows.index,list(RAW)],rtol=0,atol=0,equal_nan=True)
    require(np.array_equal(rows.canonical_smiles,full.loc[rows.index,'canonical_smiles']),'Class input graph changed')
    y=rows[list(RAW)].to_numpy(float); y[:,:5]=np.log10(y[:,:5])
    _,g=np.unique(rows.group_id,return_inverse=True)
    c=Cohort('class494',rows,y,rows.index.isin(fulltriad.structure_id),g)
    pred=read(old/'outputs/outer_predictions.csv')
    pred=pred.loc[pred.cohort.eq('class')&pred.procedure.eq('primary')]
    require(len(rows)==494,'Expected494 class cohort')
    for model in ('GraphConv','Chemprop','ChemBERTa_head'):
        arms={arm:aligned(pred.loc[pred.backbone.eq(model)&pred.arm.eq(arm)],c,SEEDS,prefix='') for arm in ('C0','C1','C2','C3')}
        ev.suite(c,model,'F_class_encoding',arms,SEEDS,[pair(a,'C0') for a in ('C1','C2','C3')],legacy='group_recomputed')
    ev.audit.append(dict(cohort='class494',structures=494,complete_triads=int(c.triads.sum()),groups=int(len(set(g))),validator='Saved outer prediction coverage, finite units, unique seed/structure keys and frozen group/fold identity; all four encodings retained. Original display uses seed SD, not bootstrap CI.'))

def new_export(ev,path):
    c=cohort('unified327',UNI/'data/release/numerical',UNI/'runtime/same_row_complete_triads.csv')
    d=read(path/'final_predictions.csv')
    required=['protocol','condition','input','penalty','structure_id','group_id','outer_fold','seed','fit_id','same_row_triad_eligible']+[prefix+e for prefix in ('q_','true_','observed_') for e in EPS]
    require(set(required).issubset(d.columns),'New export missing required prediction/identity columns')
    require(not d.duplicated(['protocol','condition','backbone','seed','structure_id']).any(),'Duplicate new prediction key')
    manifest=json.loads(record(path/'manifest.json').read_text())
    require('files' in manifest,'New manifest must include file hashes')
    for name,value in manifest['files'].items():
        record(path/name); require(HASHES[str((path/name).resolve())]==value,'New export hash mismatch')
    require('selections.json' in manifest['files'],'Selection provenance must be hash-bound')
    selections=json.loads((path/'selections.json').read_text())
    require(set(d.protocol)=={'F','T'},'Require explicit F and T prediction identities')
    for protocol,frame in d.groupby('protocol'):
        for model,sub in frame.groupby('backbone'):
            arms={condition:aligned(rows,c,SEEDS) for condition,rows in sub.groupby('condition')}
            mapping={(rows.input.iloc[0],rows.penalty.iloc[0]):name for name,rows in sub.groupby('condition')}
            require(len(mapping)==len(arms),'Ambiguous input/penalty condition identity')
            expected={(i,p) for i in ('structure','category','raw_pka','fractions') for p in ('off','positive')} if protocol=='T' else ({(i,'off') for i in ('structure','category','raw_pka','fractions')}|{(i,p) for i in ('structure','category') for p in ('shared','separate')})
            require(set(mapping)==expected,'Missing or unexpected planned condition')
            for condition,rows in sub.groupby('condition'):
                require(rows.input.nunique()==rows.penalty.nunique()==1,'Condition has inconsistent factor labels')
                require(rows.fit_id.notna().all(),'Missing final fit identity')
                for _,fitrows in rows.groupby(['seed','outer_fold']): require(fitrows.fit_id.nunique()==1,'Multiple fits in one condition/seed/fold')
            off=mapping[('structure','off')]; comps=[]
            for inp in ('category','raw_pka','fractions'): comps.append(pair(mapping[(inp,'off')],off))
            for inp in ('raw_pka','fractions'): comps.append(pair(mapping[(inp,'off')],mapping[('category','off')]))
            if protocol=='T':
                for inp in ('structure','category','raw_pka','fractions'):
                    comps.append(pair(mapping[(inp,'positive')],mapping[(inp,'off')]))
                    if inp!='structure': comps.append((f'interaction_{inp}',{mapping[(inp,'positive')]:1,mapping[(inp,'off')]:-1,mapping[('structure','positive')]:-1,off:1},off))
            else:
                for inp in ('structure','category'): comps.append(pair(mapping[(inp,'separate')],mapping[(inp,'shared')]))
                # Exact immutable F references, including fit IDs, must match old rows.
                old=read(UNI/'outputs/export/combined/final_predictions.csv')
                for inp,pen,cell in [('structure','off','00'),('category','off','10'),('structure','shared','01'),('category','shared','11')]:
                    rows=sub.loc[sub.condition.eq(mapping[(inp,pen)])].set_index(['seed','structure_id']).sort_index()
                    ref=old.loc[old.backbone.eq(model)&old.cell.eq(cell)].set_index(['seed','structure_id']).sort_index()
                    np.testing.assert_array_equal(rows.fit_id,ref.fit_id)
                    np.testing.assert_array_equal(arms[mapping[(inp,pen)]],aligned(ref.reset_index(),c,SEEDS))
            ev.suite(c,model,protocol+'_new',arms,SEEDS,comps,hard=False,legacy='group_recomputed')
            # Zero-inclusive policy is prespecified from inner scores; reuse exact
            # off/positive predictions by fold, without any held-out selection.
            zero_arms={}; zero_comps=[]
            inputs=('structure','category','raw_pka','fractions') if protocol=='T' else ('structure','category')
            for inp in inputs:
                offname=mapping[(inp,'off')]; posname=mapping[(inp,'positive' if protocol=='T' else 'separate')]
                zname=posname+'__zero_eligible'; q=arms[posname].copy()
                for fold in range(1,6):
                    entry=selections[f'{model}-outer{fold}-P{("structure","category","raw_pka","fractions").index(inp)}']
                    policy=entry if protocol=='T' else entry['R2']
                    require(policy['zero_eligible_lambda'] in (0,policy['selected_lambda']),'Invalid zero-inclusive selected lambda')
                    if policy['zero_eligible_lambda']==0:
                        mask=c.rows.outer_fold.eq(fold).to_numpy();q[:,mask]=arms[offname][:,mask]
                zero_arms.update({zname:q,offname:arms[offname],posname:arms[posname]})
                zero_comps.extend([pair(zname,offname),pair(zname,posname)])
            ev.suite(c,model,protocol+'_new_zero_policy',zero_arms,SEEDS,zero_comps,hard=False,legacy='group_recomputed')
