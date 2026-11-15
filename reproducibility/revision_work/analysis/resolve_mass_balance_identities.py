"""Apply evidence-backed dispositions for the 65 labelled identity flags.

Consumes the preceding review candidate, not live database values. Produces a
new manifest-tracked candidate; archived PK values and split assignments are
preserved. Unsupported component-specific labels are masked in model inputs.
Database agreement supports identity curation, not clinical PK-label validity.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, html
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from audit_mass_balance_data import ROOT, WORK, TASKS, canonical, structure_descriptor, sha, write_json, summarise_counts
from mass_balance_controls import residual_log10

OLD=WORK/'review_20261112/mass_balance_data_audit_20261112'
OUT=WORK/'review_20261112/mass_balance_identity_resolution_20261112'
RDLogger.DisableLog('rdApp.*')
BASE=pd.read_csv(OLD/'candidate_records_review_only.csv').set_index('ORIG_INDEX',drop=False)
q=pd.read_csv(OLD/'identity_adjudication_queue.csv');q=q[q.has_PK_label].copy()
a=pd.read_csv(OUT/'source_agreement.csv',dtype={'pubchem_CAS_exact_parent_agreement':str,'pubchem_CAS_connectivity_agreement':str}).set_index('row_idx')
assert len(q)==65 and q.row_idx.is_unique and a.index.is_unique
e=pd.read_csv(OLD/'identity_evidence_comparison.csv')
refs=json.loads((OUT/'exception_reference_index.json').read_text())
ref_by_row={rid:[x for x in refs if rid in x['row_indices']] for rid in q.row_idx}

EXCLUDE={
 785:'The e-Drug3D card cites PMID 19433570. That study pooled colistin A/B after intravenous colistimethate (CMS): 13.7 L/h is CMS clearance, whereas 189 L is formed-colistin V/fm and 14.4 h is its half-life. The archived endpoints therefore mix analytes, and none is specific to the saved A-like structure. Exclude this record; do not substitute an arbitrary component.'}
TRAMADOL_ALIASES=[495,929,930]
IVERMECTIN_MASKS={
 't1/2(hour)':'The FDA product label states 18 h for ivermectin without an explicit component qualifier. A separate H2B1a study also reports about 18 h (Guzzo et al., DOI 10.1177/009127002237994), but its doses and provenance do not establish the archived measurement or half-life phase. Mask rather than assume attribution.',
 'PPB(percentage)':'The archived 99% PPB has no source citation on the e-Drug3D card and is not component-attributed in the FDA label. Do not supervise B1a with an unverified mixture-level label.'}

SPECIAL={
171:('retain','racemate_connectivity','NIST identifies the recorded CAS as racemic phenylpropanolamine. Retain the ChEMBL-matching stereo-unspecified API and existing hydrochloride component; do not replace it with one norephedrine enantiomer. Relative erythro stereochemistry is documented metadata, not encoded by this connectivity-level graph.'),
173:('retain','supported_geometric_representation','Current full structure equals ChEMBL19 and FDA UNII W733B0S9SD (N(E) representation). PubChem omits this geometric specification; that omission is not evidence that the saved structure is wrong.'),
203:('chembl','racemate_with_fixed_ring_geometry','FDA describes atropine as a racemate. Use the ChEMBL tropine ring stereochemistry and leave the tropate side-chain centre unspecified; the old graph specified one side-chain enantiomer and omitted one ring centre.'),
259:('retain','racemate_connectivity','Pentazocine is supplied as a racemate with defined relative geometry. Retain the stereo-unspecified ChEMBL connectivity, with racemate/relative-geometry metadata, rather than substitute the single enantiomer returned by PubChem.'),
270:('clear_atom_stereo','racemate_connectivity','FDA identifies norgestrel as a racemate of levonorgestrel and its enantiomer. The old/ChEMBL graph partially fixes absolute steroid centres. Remove atom stereo tags to match the PubChem racemate connectivity; retain the documented steroid relative geometry in metadata rather than falsely assigning one absolute isomer.'),
332:('retain','stereoisomer_mixture_connectivity','A primary human PK study identifies nadolol as four stereoisomers. Retain the stereo-unspecified ChEMBL connectivity; the mixture shares one constitution. Its ring-relative geometry is metadata and is not fully represented in the model graph.'),
376:('retain','racemate_connectivity','FDA chemistry review identifies nabilone as the RR/SS racemic trans pair. Retain stereo-unspecified ChEMBL connectivity and record the trans racemate; do not use one PubChem enantiomer as the measured substance.'),
382:('retain','racemate_connectivity','The product label identifies milnacipran hydrochloride as a cis racemate. Retain the stereo-unspecified ChEMBL API and archived hydrochloride; cis relative geometry and racemic status are retained as metadata.'),
470:('retain','stereoisomer_mixture_connectivity','FDA identifies itraconazole as four stereoisomers with a shared constitution. Retain the stereo-unspecified ChEMBL-matching API; do not select one absolute stereoisomer. Existing disconnected component convention is unchanged, and this disposition validates active-moiety identity rather than a clinical salt-specific PK label.'),
483:('chembl','racemate_connectivity','FDA identifies cisapride as a cis racemate. Replace the single-enantiomer saved graph with the ChEMBL stereo-unspecified connectivity and record the cis pair. No new hydrate is inferred for the existing parent record.'),
487:('granisetron_chebi','fixed_endo_geometry','FDA specifies endo granisetron and ChEBI:5537 supplies the corresponding explicit ring/endo SMILES. Restore that geometry and retain the archived hydrochloride component. Reject the deuterated PubChem candidate despite its non-specific CAS synonym.'),
493:('retain','racemate_connectivity','Fluvastatin is the 3R,5S/3S,5R racemate. Retain ChEMBL-matching atom-stereo-unspecified connectivity and its double-bond specification, with the known relative pair recorded as metadata.'),
508:('chembl','anomer_unspecified','ChEBI:2376 and ChEMBL404271 agree on cyclic acarbose with the reducing-end anomer unspecified. Remove the unsupported fixed anomer using that shared structure; do not replace the oligosaccharide with the open-chain PubChem representation.'),
575:('retain','racemate_connectivity','USP identifies cevimeline as the cis racemate, including the archived free-base CAS. Retain stereo-unspecified ChEMBL connectivity, with cis-pair metadata, instead of the partially specified PubChem stereoisomer.'),
632:('retain','fixed_ring_geometry','Current trospium cation equals ChEMBL1888176, including the ring/ester geometry. FDA identifies the 3-alpha ester. PubChem loses the ester-bearing-centre specification; retain the supported structure.'),
709:('ivermectin_b1a','component_specific_labels_only','Retain measured analyte H2B1a, with the explicit PubChem graph corroborated by FDA UNII 91Y2202OUW. FDA reports B1a peak 46.6 ng/mL at about 4 h after 12 mg oral ivermectin mixture in fasting volunteers; with B1a MW 875.10 this matches archived 0.053 uM and 4 h after rounding. Linkage to the archived uncited card is inferred by matching value/context, not a recovered study identifier. Keep only Cmax/Tmax; mask archived half-life and PPB because component-specific provenance is unresolved. Preserve the product CAS separately from analyte CAS 71827-03-7.'),
759:('retain','fixed_ring_geometry','Current API equals ChEMBL428. The TROVAN label confirms the 1-alpha,5-alpha,6-alpha bicyclic geometry and mesylate form. Preserve this geometry and mesylate, rather than the less specified PubChem API.'),
931:('chembl','racemate_connectivity','FDA states that tramadol is administered as the (+/-)-cis racemate. Replace the saved single-enantiomer graph with the ChEMBL stereo-unspecified connectivity; record the cis racemate as metadata.'),
1093:('retain','salt_protonation_representation','FDA specifies the cationic serdexmethylphenidate chloride form and its three absolute stereocentres. Current API matches ChEMBL; PubChem uses a zwitterionic acid/base form. Retain the documented chloride representation and stereochemistry.'),
1098:('retain','supported_geometric_representation','Current structure matches ChEMBL and the FDA UNII key LPAUOXUZGSBGDU-STDDISTJSA-N for (2Z,5Z,2R) ponesimod. PubChem leaves the imine geometry unspecified. No correction is warranted.'),
1101:('chembl','fixed_cis_ring_geometry','The FDA label explicitly specifies cis pralsetinib and the S side chain. Correct the inverted cyclohexane relative centre to the ChEMBL structure. PubChem lacks the ring-relative specification and cannot settle this difference alone.'),
}

def target_smiles(rid,current):
 if rid in EXCLUDE:return None
 mode=SPECIAL.get(rid,('chembl',))[0]
 if mode=='retain':return current
 if mode=='ivermectin_b1a':
  prop=json.loads((OUT/'sources/reference_ivermectin_b1a_pubchem.json').read_text())['PropertyTable']['Properties'][0]
  assert prop['CID']==6321424 and prop['InChIKey']=='AZSNMRSAGSSBNP-XPNPUAGNSA-N'
  assert Chem.MolToInchiKey(Chem.MolFromSmiles(prop['SMILES']))==prop['InChIKey']
  assert prop['InChIKey'] in (OUT/'sources/reference_ivermectin_b1a_unii.html').read_text(encoding='utf-8')
  return prop['SMILES']
 target=a.loc[rid,'chembl_smiles']
 if mode=='clear_atom_stereo':
  m=Chem.MolFromSmiles(current)
  for atom in m.GetAtoms():atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
  return Chem.MolToSmiles(m)
 if mode=='granisetron_chebi':
  return 'CN1[C@@H]2CCC[C@H]1C[C@@H](NC(=O)c1nn(C)c3ccccc13)C2.Cl'
 if rid==459:
  # Keep the originally represented sodium carboxylate; restore P stereo only.
  m=Chem.MolFromSmiles(target)
  matches=m.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1]'))
  assert len(matches)==1
  oxygen=m.GetAtomWithIdx(matches[0][2]);oxygen.SetFormalCharge(-1);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(True)
  Chem.SanitizeMol(m)
  return Chem.MolToSmiles(m)+'.[Na+]'
 if rid==1103:return target+'.Cl.Cl'  # Label-supported dihydrochloride.
 assert '.' not in current, f'Unreviewed component handling {rid}'
 return target

def build():
 d=BASE.copy(deep=True);ledger=[]
 for r in q.itertuples():
  rid=int(r.row_idx);old=d.loc[rid,'canonical_smiles'];new=target_smiles(rid,old)
  rec=a.loc[rid]; chembl_url=f'https://www.ebi.ac.uk/chembl/api/data/molecule/{rec.chembl_id}.json'
  references=[chembl_url]
  for x in ref_by_row[rid]:
   assert isinstance(x['cache'],str),f'Missing primary source {x}'
   references.append(x['url'])
  if rid in EXCLUDE:action='exclude';category='unresolved_component_to_PK_mapping';reason=EXCLUDE[rid]
  elif rid in SPECIAL:
   _,category,reason=SPECIAL[rid];action='correct' if canonical(old)!=canonical(new) else 'retain'
  else:
   assert pd.notna(rec.pubchem_CAS_exact_parent_agreement),rid
   cid=int(rec.pubchem_CAS_exact_parent_agreement.split(',')[0])
   references.append(f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}')
   action='correct' if canonical(old)!=canonical(new) else 'retain'
   category='corroborated_stereo_or_isotope_correction'
   reason='The explicitly named ChEMBL structure and a PubChem structure carrying the exact archived CAS agree after parent/tautomer comparison. Use their shared stereochemical/isotopic identity, correcting missing or conflicting specifications in the archived graph.'
   if rid==391:category='racemate_connectivity';reason+=' Nimodipine is stereo-unspecified in both sources; remove the unsupported single-enantiomer assignment.'
   if rid==459:reason+=' Preserve the sodium carboxylate; the Health Canada monograph supports the phosphorus/side-chain stereochemistry.'
   if rid==1103:reason+=' Preserve both HCl components; the FDA label supports the R drug as the dihydrochloride.'
   if rid==1134:reason+=' Restore the explicitly deuterated methyl group; ordinary methyl is not the same isotopic drug.'
  new=canonical(new) if new else None
  if new:
   assert Chem.MolFromSmiles(new) is not None
   # These 65 cases do not authorise changing the active-moiety constitution.
   od=structure_descriptor(old);nd=structure_descriptor(new)
   assert od['connectivity']==nd['connectivity'],(rid,'connectivity changed')
   assert Chem.GetFormalCharge(Chem.MolFromSmiles(old))==Chem.GetFormalCharge(Chem.MolFromSmiles(new)),(rid,'charge changed')
   assert len(Chem.GetMolFrags(Chem.MolFromSmiles(old)))==len(Chem.GetMolFrags(Chem.MolFromSmiles(new))),(rid,'fragment count changed')
   if rid not in SPECIAL:
    assert nd['parent_key']==rec.chembl_parent_key,(rid,'target does not match corroborated identity')
   d.loc[rid,'canonical_smiles']=new
  d.loc[rid,'pending_identity']=False
  ledger.append(dict(row_idx=rid,Name=r.Name,CAS=d.loc[rid,'CAS'],action=action,category=category,reason=reason,old_smiles=old,new_smiles=new,old_inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles(old)),new_inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles(new)) if new else None,chembl_id=rec.chembl_id,references=' | '.join(references),archived_PK_values_changed=False,model_endpoints_masked=json.dumps(list(IVERMECTIN_MASKS) if rid==709 else []),historical_partition_changed=False))
 l=pd.DataFrame(ledger); assert len(l)==65 and l.row_idx.is_unique
 d['identity_status']=d.index.map(l.set_index('row_idx').action).fillna('unchanged_by_this_65_record_review')
 d['identity_exclusion']=d.index.isin(EXCLUDE)
 # Correcting only row 931 would split one old deduplicated tramadol unit into
 # two pseudo-independent units. The four source rows carry IDENTICAL PK values,
 # which the source product card reports for marketed tramadol, not each isomer.
 # Keep the racemate row and retain the other CAS records as provenance aliases.
 aliases=[]
 for rid in TRAMADOL_ALIASES:
  assert BASE.loc[rid,'Name']=='TRAMADOL'
  assert BASE.loc[rid,TASKS].equals(BASE.loc[931,TASKS])
  assert BASE.loc[rid,'canonical_smiles']==BASE.loc[931,'canonical_smiles']
  aliases.append({'row_idx':rid,'Name':'TRAMADOL','original_CAS':BASE.loc[rid,'CAS'],'modelled_representative_row':931,'representative_CAS':BASE.loc[931,'CAS'],'action':'provenance_alias_only','reason':'Identical seven-endpoint vector is reported for the marketed tramadol product; no independent isomer-specific PK evidence. Retain once under the supported racemate record, without changing the archived CAS or target values.','reference':'https://chemoinfo.ipmc.cnrs.fr/MOLDB/CARD/edrug3d_714_1674_1676_1678_1680_1675_1677_1679.html | https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/020281s040lbl.pdf'})
 d['provenance_alias_only']=d.index.isin(TRAMADOL_ALIASES)
 d['modelled_analyte_name']=d.Name
 d['modelled_analyte_CAS']=d.CAS
 d['administration_context']='not_reaudited_in_this_identity_pass'
 d['masked_model_endpoints']='[]'
 d.loc[709,'modelled_analyte_name']='IVERMECTIN B1A (H2B1a)'
 d.loc[709,'modelled_analyte_CAS']='71827-03-7'
 d.loc[709,'administration_context']='12 mg oral ivermectin B1a/B1b mixture; fasting healthy volunteers; Cmax/Tmax measured for H2B1a'
 d.loc[709,'masked_model_endpoints']=json.dumps(list(IVERMECTIN_MASKS))
 assert d[TASKS].equals(BASE[TASKS]) and d.partition.equals(BASE.partition)
 unchanged=~d.index.isin(q.row_idx)
 assert d.loc[unchanged,'canonical_smiles'].equals(BASE.loc[unchanged,'canonical_smiles'])
 d.to_csv(OUT/'all_records_after_identity_review.csv',index=False)
 l.to_csv(OUT/'identity_resolution_65.csv',index=False)
 l[l.action.eq('exclude')].to_csv(OUT/'identity_exclusions.csv',index=False)
 l[l.action.eq('correct')].to_csv(OUT/'identity_corrections.csv',index=False)
 pd.DataFrame(aliases).to_csv(OUT/'tramadol_source_aliases.csv',index=False)
 masks=[{'row_idx':709,'source_Name':d.loc[709,'Name'],'modelled_analyte_name':d.loc[709,'modelled_analyte_name'],'endpoint':c,'archived_value':d.loc[709,c],'model_value':np.nan,'reason':why,'references':'https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/050742s030lbl.pdf | https://chemoinfo.ipmc.cnrs.fr/MOLDB/CARD/edrug3d_1068.html'} for c,why in IVERMECTIN_MASKS.items()]
 pd.DataFrame(masks).to_csv(OUT/'endpoint_identity_masks.csv',index=False)

 valid=d[d.canonical_smiles.notna() & d.canonical_smiles.ne('') & ~d.identity_exclusion & ~d.provenance_alias_only].copy()
 for c in IVERMECTIN_MASKS:valid.loc[709,c]=np.nan
 assert valid.drop(index=709)[TASKS].equals(d.loc[valid.index.drop(709),TASKS])
 assert valid.loc[709,['Cmax_uM','Tmax']].equals(BASE.loc[709,['Cmax_uM','Tmax']])
 units=[];mapping=[];triads=[];duplicate=[]
 for smi,g in valid.groupby('canonical_smiles',sort=True):
  conflict=[c for c in TASKS if g[c].nunique(dropna=True)>1]
  assert not conflict,(g.Name.tolist(),conflict)
  sid=hashlib.sha256(smi.encode()).hexdigest()[:16]
  if len(g)>1:duplicate.append({'structure_id':sid,'rows':json.dumps(g.index.tolist()),'names':' | '.join(g.Name),'historical_partitions':' | '.join(sorted(g.partition.unique()))})
  row={'structure_id':sid,'canonical_smiles':smi,'source_rows':json.dumps(g.index.tolist()),'Name':' | '.join(g.Name),'historical_partition':' | '.join(sorted(g.partition.unique())),'pending_identity':bool(g.pending_identity.any()),'volume_definition':'source_confirmed_central' if g.volume_definition.eq('source_confirmed_central').any() else 'unclassified'}
  for field in ['modelled_analyte_name','modelled_analyte_CAS','administration_context','masked_model_endpoints']:row[field]=' | '.join(g[field].astype(str).unique())
  for c in TASKS:row[c]=g[c].dropna().iloc[0] if g[c].notna().any() else np.nan
  units.append(row)
  for rid,r in g.iterrows():
   mapping.append({'row_idx':rid,'structure_id':sid,'historical_partition':r.partition})
   if r[TASKS[:3]].notna().all():triads.append({'structure_id':sid,'row_idx':rid,'Name':r.Name,**{c:r[c] for c in TASKS[:3]}})
 u=pd.DataFrame(units);labelled=u[u[TASKS].notna().any(axis=1)].copy()
 assert not labelled.pending_identity.any()
 labelled.to_csv(OUT/'mass_balance_labelled_structures.csv',index=False)
 tramadol_sid=hashlib.sha256(d.loc[931,'canonical_smiles'].encode()).hexdigest()[:16]
 for rid in TRAMADOL_ALIASES:mapping.append({'row_idx':rid,'structure_id':tramadol_sid,'historical_partition':d.loc[rid,'partition'],'mapping_role':'provenance_alias_only'})
 pd.DataFrame(mapping).to_csv(OUT/'row_to_structure.csv',index=False)
 pd.DataFrame(duplicate).to_csv(OUT/'duplicate_groups_after_identity.csv',index=False)
 counts=summarise_counts(labelled.rename(columns={'historical_partition':'partition'}));counts.to_csv(OUT/'endpoint_counts.csv',index=False)
 t=pd.DataFrame(triads).drop_duplicates(['structure_id',*TASKS[:3]])
 assert not t.structure_id.duplicated().any()
 t['signed_residual_log10']=residual_log10(np.log10(t[TASKS[:3]].to_numpy(float)))
 t.to_csv(OUT/'same_row_complete_triads.csv',index=False)
 summary={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'65 labelled identity flags and necessary duplicate provenance reconciliation; no split redesign or PK-label changes','reviewed_labelled_flags':len(l),'dispositions':l.action.value_counts().to_dict(),'retained_after_review':int(l.action.ne('exclude').sum()),'excluded_names':l[l.action.eq('exclude')].Name.tolist(),'labelled_structures_before':953,'labelled_structures_after':len(labelled),'identity_exclusion_percent_of_previous_labelled':100*(953-len(labelled))/953,'same_row_complete_triads':len(t),'unique_valid_structures_after_identity_exclusions':len(u),'entirely_unlabelled_structures':int((~u[TASKS].notna().any(axis=1)).sum()),'remaining_labelled_identity_flags':int(labelled.pending_identity.sum()),'unlabelled_flags_not_in_scope':6,'endpoint_counts':{r.endpoint:int(r.observed) for r in counts[counts.partition.eq('all')].itertuples()},'historical_partition_counts':labelled.historical_partition.value_counts().to_dict(),'historical_partition_changed':False,'new_splits_created':False,'remaining_exact_duplicate_groups':len(duplicate),'tramadol_provenance_aliases':3,'tramadol_modelled_units':1,'PK_labels_changed':False,'training_ready':False,'remaining_work':['separate review/freeze status of earlier 59 staged connectivity corrections remains recorded','split design and grouping review','scientific scope and runner validation'],'representation_caveat':'Known racemates/stereoisomer mixtures with shared constitution may be represented by stereo-unspecified graphs. Known relative configurations are documented in the ledger; this does not claim the graph encodes mixture composition or all relative stereochemistry. Different-connectivity mixtures are excluded when PK cannot be assigned to one component.'}
 summary.update(scope='65 labelled identity flags, necessary duplicate reconciliation, and two explicit component-attribution endpoint masks; no split redesign',archived_PK_values_changed=False,model_endpoint_masks=len(masks),partially_retained_records=['IVERMECTIN: H2B1a Cmax/Tmax only'])
 summary.pop('PK_labels_changed')
 write_json(OUT/'resolution_summary.json',summary)
 manifest={'inputs':{str(p):sha(p) for p in [OLD/'candidate_records_review_only.csv',OLD/'identity_adjudication_queue.csv',OUT/'source_agreement.csv',Path(__file__)]},'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'},'training_ready':False}
 write_json(OUT/'manifest.json',manifest)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':build()
