"""Ionisation schematic, reference accuracy and five single-pH comparisons."""

import argparse
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
OLD=EXP/'outputs/scoring'
V2=HERE.parent/'ionisation_main_v2'
SINGLE_PH_RENDER=HERE.parent/'single_ph_v2/render.py'
sys.path.insert(0, str(HERE.parent))
from ionisation_style import AXES, CATEGORIES, CONTROL, FONT, GRID, INK, MODELS as SHARED_MODELS, MUTED, ROW, ENDPOINT_COLORS
from schematic import draw_schematic

SCHEMATIC_SOURCE = HERE / "schematic.py"
MODELS = {**SHARED_MODELS, "ChemBERTa_head": "ChemBERTa"}
MODEL_LEFT = .242
MODEL_STEP = .244
MODEL_WIDTH = .209
SCATTER_LEFT = .095
SCATTER_STEP = .30
SCATTER_WIDTH = .27
MAE_LABEL = "Change in MAE vs structure only (%)"
ENDPOINTS={
    't_half':('Half-life',ENDPOINT_COLORS["t_half"],316),
    'Vd':('Volume of distribution',ENDPOINT_COLORS["Vd"],259),
    'CL':('Clearance',ENDPOINT_COLORS["CL"],246),
    'Cmax':('Peak concentration',ENDPOINT_COLORS["Cmax"],190),
    'Tmax':('Time to peak',ENDPOINT_COLORS["Tmax"],249),
    'F':('Bioavailability',ENDPOINT_COLORS["F"],172),
    'PPB':('Protein binding',ENDPOINT_COLORS["PPB"],292),
}
SEEDS=[101,202,303,404,505,606,707,808,909]
JITTER=np.array([-.105,.085,-.05,.12,.015,-.08,.055,-.015,.09])
SOURCES={}
FIGURE_WIDTH = 7.0
FIGURE_HEIGHT = 9.15
SCATTERS=EXP/'figures/ionisation_observed_predicted_v1'
PH_ARMS={'PH2':'pH 2.0', 'PH5':'pH 5.0', 'PH65':'pH 6.5', 'REF74':'pH 7.4', 'PH8':'pH 8.0'}
plt.rcParams.update({'font.family':FONT,'font.size':8.5,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':AXES,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.linewidth':.7})


def read_csv(path,check_manifest=None):
    digest=sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path.relative_to(EXP))]=digest
    if check_manifest:
        manifest=json.loads(check_manifest.read_text())
        assert manifest['files'][path.name]==digest,path.name
    return pd.read_csv(path)


def subset(d,**criteria):
    mask=pd.Series(True,index=d.index)
    for k,v in criteria.items():mask &= d[k].eq(v)
    return d.loc[mask]


def one(d,**criteria):
    x=subset(d,**criteria)
    assert len(x)==1,(criteria,len(x))
    return x.iloc[0]


def heading(fig, letter, height):
    fig.text(.025, height/FIGURE_HEIGHT, letter, fontsize=12, weight="bold", va="baseline")


def model_axes(fig, column, y, height):
    return fig.add_axes([MODEL_LEFT + column*MODEL_STEP, y/FIGURE_HEIGHT,
                         MODEL_WIDTH, height/FIGURE_HEIGHT])


def schematic(fig):
    heading(fig, "A", 8.94)
    ax = fig.add_axes([.025, 7.40/FIGURE_HEIGHT, .95, 1.47/FIGURE_HEIGHT])
    draw_schematic(ax)


def ionisation_performance(fig,contrasts,seeds,averages,average_seeds,ph_estimates):
    heading(fig, 'B', 7.15)
    rows=subset(contrasts,cohort='numerical',procedure='primary',arm='P1',comparator_arm='P0',scope='all',stratum='all').copy()
    legacy_rows=rows.copy()
    for idx,r in rows.iterrows():
        direct=one(ph_estimates,arm='REF74',backbone=r.backbone,endpoint=r.endpoint)
        assert np.isclose(direct.relative_delta_pct,r.relative_delta_pct)
        rows.loc[idx,'crossed95_low']=direct.crossed95_low_pct*r.comparator_mae/100
        rows.loc[idx,'crossed95_high']=direct.crossed95_high_pct*r.comparator_mae/100
    ss=subset(seeds,cohort='numerical',procedure='primary',arm='P1',comparator_arm='P0',scope='all',stratum='all')
    assert len(rows)==21 and len(ss)==189
    plotted=[];all_values=[]
    for _,r in rows.iterrows():
        s=subset(ss,backbone=r.backbone,endpoint=r.endpoint).sort_values('seed');assert s.seed.tolist()==SEEDS
        fac=100/r.comparator_mae;values=fac*s.delta.to_numpy()
        all_values.extend([*values,fac*r.crossed95_low,fac*r.crossed95_high])
        for seed,value in zip(SEEDS,values):plotted.append({'backbone':r.backbone,'endpoint':r.endpoint,'seed':seed,'relative_delta_pct':value})
    all_values.extend(average_seeds.relative_delta_pct)
    all_values.extend(averages.crossed95_low_pct)
    all_values.extend(averages.crossed95_high_pct)
    left=min(-20.,5*np.floor(min(all_values)/5));right=max(5.,5*np.ceil(max(all_values)/5))
    for j,(model,name) in enumerate(MODELS.items()):
        ax=model_axes(fig, j, 5.64, 1.40)
        for y in [1,2]:ax.axhspan(y-.43,y+.43,color=ROW,zorder=0)
        ax.axhspan(6.65,7.43,color=ROW,zorder=0)
        ax.axhline(6.55,color='#CCD5DA',lw=.6)
        ax.axvline(0,color='#8B9AA5',lw=.8,ls='--',zorder=1);ax.grid(axis='x',color=GRID,lw=.6,zorder=0)
        for y,(ep,(_,color,n)) in enumerate(ENDPOINTS.items()):
            r=one(rows,backbone=model,endpoint=ep);assert r.n_structures==n
            s=subset(ss,backbone=model,endpoint=ep).sort_values('seed');factor=100/r.comparator_mae
            ax.scatter(s.delta.to_numpy()*factor,y+JITTER,s=10,color=color,alpha=.6,edgecolor='none',zorder=2)
            ax.plot(factor*np.array([r.crossed95_low,r.crossed95_high]),[y,y],color=color,lw=1.15,zorder=3)
            ax.plot(r.relative_delta_pct,y,'D',ms=4,color=color,mec=INK,mew=.5,zorder=4)
        avg=one(averages,backbone=model)
        avg_seeds=subset(average_seeds,backbone=model).sort_values('seed')
        assert avg_seeds.seed.tolist()==SEEDS
        expected=pd.DataFrame(plotted).query('backbone == @model').groupby('seed').relative_delta_pct.mean().reindex(SEEDS)
        assert np.allclose(expected,avg_seeds.relative_delta_pct)
        ax.scatter(avg_seeds.relative_delta_pct,7+JITTER,s=10,color=CONTROL,alpha=.65,edgecolor='none',zorder=2)
        ax.plot([avg.crossed95_low_pct,avg.crossed95_high_pct],[7,7],color=CONTROL,lw=1.2,zorder=3)
        ax.plot(avg.relative_delta_pct,7,'D',ms=4.3,color=CONTROL,mec=INK,mew=.5,zorder=4)
        ax.set(xlim=(left,right),ylim=(7.65,-.6),yticks=range(8))
        labels=[label for label,_,_ in ENDPOINTS.values()]+['Average']
        ax.set_yticklabels(labels if j==0 else [],fontsize=8.2)
        if j == 0:
            ax.get_yticklabels()[-1].set_weight('bold')
        ax.set_title(name,fontsize=8.8,weight='bold',pad=6)
        ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5));ax.tick_params(axis='x',labelsize=7.8)
    fig.text(.597, 5.32/FIGURE_HEIGHT, MAE_LABEL, ha='center', fontsize=8.5)
    legacy_rows.to_csv(HERE/'panel_b_estimates.csv',index=False)
    rows.to_csv(HERE/'panel_b_displayed_estimates.csv',index=False)
    pd.DataFrame(plotted).to_csv(HERE/'panel_b_seeds.csv',index=False)


def vd_scatter_row(fig):
    manifest=json.loads((SCATTERS/'manifest.json').read_text())
    for name in ['plotted_points.csv','metrics.csv']:
        assert sha256((SCATTERS/name).read_bytes()).hexdigest()==manifest['outputs'][name]
    data=read_csv(SCATTERS/'plotted_points.csv')
    data=subset(data,arm='REF74',endpoint='Vd')
    metrics=read_csv(SCATTERS/'metrics.csv')
    heading(fig, "C", 5.07)
    handles = [Line2D([], [], marker="o", mfc=colour, mec="none", ls="none", ms=4.4, label=label)
               for label, colour, _ in CATEGORIES.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.076, 5.19/FIGURE_HEIGHT),
               ncol=4, frameon=False, fontsize=8, handletextpad=.3, columnspacing=1.1, borderaxespad=0)
    lo,hi=manifest['axis_limits_log10']['Vd']
    for j,(model,name) in enumerate(MODELS.items()):
        ax = fig.add_axes([SCATTER_LEFT + j*SCATTER_STEP, 2.64/FIGURE_HEIGHT,
                           SCATTER_WIDTH, SCATTER_WIDTH*FIGURE_WIDTH/FIGURE_HEIGHT])
        d=subset(data,backbone=model);assert len(d)==259
        r=one(metrics,arm='REF74',endpoint='Vd',backbone=model)
        y,x=d.observed_log10.to_numpy(),d.predicted_log10.to_numpy()
        assert np.isclose(1-((y-x)**2).sum()/((y-y.mean())**2).sum(),r.r2_displayed_seed_mean)
        ax.plot([lo,hi],[lo,hi],color='#9AA5AD',ls=(0,(4,3)),lw=.7,zorder=1)
        for cat, (_, colour, _) in CATEGORIES.items():
            c=subset(d,ionisation_class=cat)
            ax.scatter(c.predicted_log10,c.observed_log10,s=11,facecolor=colour,marker="o",alpha=.78,
                       edgecolors="none",linewidths=0,zorder=3)
        ax.set(xlim=(lo,hi),ylim=(lo,hi));ax.set_aspect('equal',adjustable='box')
        ax.spines[['top','right']].set_visible(False)
        ax.set_xticks([1,2,3,4,5]);ax.set_yticks([1,2,3,4,5]);ax.tick_params(labelsize=7.5)
        ax.grid(color=GRID,lw=.5)
        ax.set_title(name+'\n'+r'$R^2$'+f' = {r.r2_displayed_seed_mean:.3f}',
                     fontsize=8.2,weight='bold',pad=7)
        ax.set_xlabel(r'Predicted $\log_{10} V_d$ (L)',fontsize=8.0,labelpad=3)
        if j==0:ax.set_ylabel(r'Observed $\log_{10} V_d$ (L)',fontsize=8.0,labelpad=4)
    data.to_csv(HERE/'panel_b_vd_scatter_points.csv',index=False)


CAPTION = '''Effects of ionisation inputs on PK prediction.

A, Molecular structural representations are concatenated with a categorical ionisation input before the prediction head. The four operational categories are acid-only, base-only, neither-active and dual-active, defined using acidic and basic pKa thresholds at the selected pH. Acid-only and base-only indicate anionic-like and cationic-like character, respectively. Dual-active can be consistent with zwitterionic character, and assignment to a specific microspecies requires additional information. Each model receives one four-channel one-hot vector at pH 2.0, 5.0, 6.5, 7.4 or 8.0. Empty cells indicate the four input dimensions. Marker colours identify the four operational categories. Output symbol colours identify the seven PK endpoints. Categories differ from their pH 7.4 assignment for 206, 85, 38 and 32 of the 327 structures at pH 2.0, 5.0, 6.5 and 8.0, respectively.

B, Adding the pH 7.4 category to structure-only models on the numerical-pKa cohort. Dots show nine paired seed differences in held-out MAE. Diamonds are means and bars are exploratory 95% crossed seed/group bootstrap intervals. All observed endpoint labels enter their respective comparisons. Label counts are 316 for half-life, 259 for volume of distribution, 246 for clearance, 190 for peak concentration, 249 for time to peak, 172 for bioavailability and 292 for protein binding. Estimate colours match the output symbols in A. Black row labels identify the endpoints, with shading highlighting Vd and clearance. Five positive endpoints use log10 MAE; F and PPB use percentage-point MAE, retaining zero labels. Each absolute paired change and interval is divided by the fixed nine-seed mean comparator MAE. The Average row gives equal weight to the seven endpoint percentage changes within each seed, followed by the mean across seeds. Its interval resamples groups and seed trajectories jointly across all seven endpoints, preserving their dependence and using each endpoint's resampled labelled count. Average changes are -3.79%, -5.47% and -1.49% for GraphConv, Chemprop and frozen ChemBERTa, respectively. These numerical-cohort comparisons are secondary analyses of the ionisation representation experiment. The category-encoding comparison in the 494-structure class cohort appears in panel C of the ionisation supplementary figure. The main figure uses its 327-structure subset throughout.

C, Observed versus predicted Vd for the 259 labelled structures, with the pH 7.4 category supplied to each model. Horizontal coordinates are mean held-out log10 predictions across nine training seeds, corresponding to geometric mean predictions on the litre scale. Vertical coordinates are observed log10 Vd. R-squared is computed from the displayed averages as 1 - sum((observed - predicted)^2)/sum((observed - mean(observed))^2). The MAE comparisons use scores computed separately for the nine seeds. Circle colours indicate the operational pH 7.4 category: navy for acid-only, cyan for base-only, yellow for neither-active and coral for dual-active. All three models use identical axes and all observed labels are retained. The dashed line denotes equality. Matched structure-only and ionisation scatter plots for Vd and clearance appear in panels A and B of the ionisation supplementary figure.

D, Paired changes in held-out MAE for volume of distribution (Vd, 259 labelled structures) and clearance (CL, 246 labelled structures). Each row compares a single four-channel ionisation category at pH 2.0, 5.0, 6.5, 7.4 or 8.0 with the same backbone’s structure-only model. The pH 7.4 and structure-only conditions reuse the original fitted predictions. Small dots show the nine seed differences. Diamonds show means and bars show exploratory 95% crossed seed/group bootstrap intervals. Changes and interval bounds are scaled by the fixed nine-seed mean comparator MAE for the same backbone and endpoint. Vd and CL errors are scored in log10 units. Estimate colours match their output symbols in A. pH specifies which category is supplied to the model.

Predictions are pooled across five parent/scaffold-grouped held-out outer folds within each seed. All panels use the same 327 structures in 238 groups with nine paired seeds. Bootstrap intervals use 10,000 draws of whole groups and whole seed trajectories, conditional on the observed data, folds and selected settings. The ChemBERTa transformer is frozen and its MLP head is trained. Supervised endpoint weights are all one and the mass-balance penalty is zero. Numeric pH conditions define input features for prediction of the original PK endpoints.

Approximate physiological examples for the tested pHs include the gastric lumen at pH 2.0 and the proximal small-intestinal lumen at pH 6.5 (Evans et al., 1988), the acidic lysosomal lumen at pH 5.0 (Johnson et al., 2016), arterial blood at pH 7.4 (De Keijzer et al., 2023), and the mitochondrial matrix in cultured cells at pH 8.0 (Llopis et al., 1998). The experiments vary the ionisation category supplied to the model while retaining the original observed PK labels. Compartment-specific PK and ion trapping were outside the scope of these experiments.

Evans et al. (1988) https://doi.org/10.1136/gut.29.8.1035
Johnson et al. (2016) https://doi.org/10.1083/jcb.201507112
De Keijzer et al. (2023) https://doi.org/10.1007/s10877-022-00944-9
Llopis et al. (1998) https://doi.org/10.1073/pnas.95.12.6803
'''


def paired_single_ph():
    """Load the validated merged single-pH score tables."""
    spec = importlib.util.spec_from_file_location("single_ph_plot_source", SINGLE_PH_RENDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load single-pH scoring reader at {SINGLE_PH_RENDER}")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    estimates, seeds, metadata, sources = reader.checked_tables()
    if len(estimates) != 189 or len(seeds) != 1701:
        raise ValueError("Expected the complete five-pH comparison data")
    keys = ["arm", "backbone", "endpoint"]
    keep = estimates.endpoint.isin(("Vd", "CL")) & estimates.comparator_arm.eq("REF0")
    keep_seeds = seeds.endpoint.isin(("Vd", "CL")) & seeds.comparator_arm.eq("REF0")
    rows = estimates.loc[keep].copy()
    points = seeds.loc[keep_seeds].copy()
    expected = {(arm, model, endpoint) for arm in PH_ARMS for model in MODELS for endpoint in ("Vd", "CL")}
    if len(rows) != len(expected) or set(rows[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Panel D requires Vd and clearance estimates for all five pHs and three models")
    expected_seeds = {(*key, seed) for key in expected for seed in SEEDS}
    if len(points) != len(expected_seeds) or set(points[[*keys, "seed"]].itertuples(index=False, name=None)) != expected_seeds:
        raise ValueError("Panel D requires all nine paired seeds for each estimate")
    if not rows.n_structures.eq(rows.endpoint.map({ep: ENDPOINTS[ep][2] for ep in ("Vd", "CL")})).all():
        raise ValueError("Unexpected labelled structure count in panel D")
    fields = ["relative_delta_pct", "crossed95_low_pct", "crossed95_high_pct"]
    if not np.isfinite(rows[fields]).all().all() or not np.isfinite(points.relative_delta_pct).all():
        raise ValueError("Panel D contains nonfinite estimates or seed values")
    np.testing.assert_allclose(rows.set_index(keys).relative_delta_pct.sort_index(),
                               points.groupby(keys).relative_delta_pct.mean(), atol=1e-12)
    return rows, points, metadata, sources


def category_switch_counts():
    """Validate both feature ledgers and count changes from the pH 7.4 category."""
    ledger_path = EXP/"runtime/ph_category_features/ledger.csv"
    ledger_manifest = ledger_path.with_name("manifest.json")
    ledger = read_csv(ledger_path, ledger_manifest).set_index("structure_id", verify_integrity=True)
    if len(ledger) != 327 or ledger.group_id.nunique() != 238:
        raise ValueError("Unexpected schematic ledger coverage")
    dataset_path = EXP/"data/release/numerical/model_dataset.csv"
    release_manifest = dataset_path.with_name("manifest.json")
    dataset = read_csv(dataset_path, release_manifest).set_index("structure_id", verify_integrity=True)
    if set(dataset.index) != set(ledger.index):
        raise ValueError("Feature ledger does not match the numerical release")
    audited = dataset.loc[ledger.index, "ionisation_class"].map({
        "anionic": 0, "cationic": 1, "neutral": 2, "zwitterionic": 3,
    })
    if not audited.eq(ledger["code_7.4"]).all():
        raise ValueError("Feature ledger does not match the audited pH 7.4 codes")

    codes_path = EXP/"runtime/ph8_features/codes.json"
    ph8_manifest = codes_path.with_name("manifest.json")
    metadata = json.loads(ph8_manifest.read_text(encoding="utf-8"))
    if (sha256(codes_path.read_bytes()).hexdigest() != metadata["codes_sha256"]
            or sha256(dataset_path.read_bytes()).hexdigest() != metadata["dataset_sha256"]
            or metadata["release_fingerprint"] != json.loads(release_manifest.read_text())["fingerprint"]
            or metadata["ph"] != "8.0" or metadata["structure_count"] != len(ledger)):
        raise ValueError("pH 8.0 category data does not match the frozen release")
    codes = pd.Series(json.loads(codes_path.read_text(encoding="utf-8")), dtype=int)
    if set(codes.index) != set(ledger.index) or not codes.isin(range(4)).all():
        raise ValueError("Unexpected pH 8.0 category codes")
    counts = {ph: int(ledger[f"code_{ph}"].ne(ledger["code_7.4"]).sum()) for ph in ("2.0", "5.0", "6.5")}
    counts["8.0"] = int(codes.loc[ledger.index].ne(audited).sum())
    if counts != {"2.0": 206, "5.0": 85, "6.5": 38, "8.0": 32} or counts["8.0"] != metadata["switchers_vs_74"]:
        raise ValueError("Unexpected category switch counts")
    for path in (ledger_manifest, release_manifest, ph8_manifest, codes_path):
        SOURCES[path.relative_to(EXP).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return counts


def single_ph_panel(fig, estimates, seeds):
    """Display Vd and clearance on separate rows with one shared effect scale."""
    heading(fig, "D", 2.15)
    values = np.r_[0, estimates.relative_delta_pct, estimates.crossed95_low_pct,
                   estimates.crossed95_high_pct, seeds.relative_delta_pct]
    low, high = values.min(), values.max()
    pad = max(.45, .055*(high-low))
    ticks = plt.MaxNLocator(5).tick_values(low-pad, high+pad)
    xlim = (float(ticks[0]), float(ticks[-1]))
    for row, ep in enumerate(("Vd", "CL")):
        ep_label, color, count = ENDPOINTS[ep]
        for j, (model, name) in enumerate(MODELS.items()):
            ax = model_axes(fig, j, (1.22, .43)[row], .66)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=GRID, lw=.65)
            ax.axvline(0, color=AXES, lw=.8, linestyle=(0, (4, 3)), zorder=1)
            if row == 0:
                ax.set_title(name, fontsize=8.8, weight="bold", pad=6)
            if j == 0:
                label = r"$V_d$" if ep == "Vd" else ep_label
                ax.text(-.83, .5, label, transform=ax.transAxes,
                        fontsize=8.3, weight="bold", va="center", color=color)
            for y, arm in enumerate(PH_ARMS):
                if y % 2:
                    ax.axhspan(y-.48, y+.48, color=ROW, zorder=0)
                r = one(estimates, arm=arm, backbone=model, endpoint=ep)
                points = subset(seeds, arm=arm, backbone=model, endpoint=ep).sort_values("seed")
                ax.scatter(points.relative_delta_pct, y+JITTER*.77, s=6, color=color,
                           alpha=.55, edgecolor="none", zorder=2)
                ax.hlines(y, r.crossed95_low_pct, r.crossed95_high_pct, color=color, lw=1.0, zorder=3)
                ax.plot(r.relative_delta_pct, y, marker="D", color=color, ms=3.5, mec=INK, mew=.35, zorder=4)
            ax.set(xlim=xlim, ylim=(4.55, -.55), xticks=ticks, yticks=range(len(PH_ARMS)))
            ax.set_yticklabels(list(PH_ARMS.values()) if j == 0 else [], fontsize=7.8)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", labelsize=7.8, length=3, labelbottom=row == 1)
            ax.spines["left"].set_visible(False)
            if row == 0:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)
    fig.text(.597, .095/FIGURE_HEIGHT, MAE_LABEL, ha="center", fontsize=8.5)
    estimates.to_csv(HERE/"panel_c_estimates.csv", index=False)
    seeds.to_csv(HERE/"panel_c_seeds.csv", index=False)
    return xlim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify complete inputs without rendering")
    args = parser.parse_args()
    estimates, paired_seeds, new_metadata, new_sources = paired_single_ph()
    old_manifest = OLD/"scoring_manifest.json"
    average_manifest = V2/"panel_b_average_manifest.json"
    b_manifest = V2/"panel_ph_vs_baseline_manifest.json"
    old = read_csv(OLD/"contrasts.csv", old_manifest)
    old_seeds = read_csv(OLD/"contrast_seeds.csv", old_manifest)
    averages = read_csv(V2/"panel_b_average_estimates.csv", average_manifest)
    average_seeds = read_csv(V2/"panel_b_average_seeds.csv", average_manifest)
    panel_b_inputs = read_csv(V2/"panel_ph_vs_baseline_estimates.csv", b_manifest)
    counts = category_switch_counts()
    for path in (old_manifest, average_manifest, b_manifest, SCATTERS/"manifest.json", SINGLE_PH_RENDER, SCHEMATIC_SOURCE, HERE.parent/'ionisation_style.py'):
        SOURCES[path.relative_to(EXP).as_posix()] = sha256(path.read_bytes()).hexdigest()
    if args.check:
        print(f"Verified 30 panel D estimates and {len(paired_seeds)} paired seed values.")
        return
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    schematic(fig)
    pd.DataFrame([{"pH": ph, "changed_category": n, "n": 327} for ph, n in counts.items()]).to_csv(
        HERE/"panel_a_counts.csv", index=False,
    )
    ionisation_performance(fig, old, old_seeds, averages, average_seeds, panel_b_inputs)
    vd_scatter_row(fig)
    limits = single_ph_panel(fig, estimates, paired_seeds)
    schematic_ax, *plots = fig.axes
    endpoint_colours = {to_rgb(colour) for colour in ENDPOINT_COLORS.values()}
    output_symbols = {label for label in schematic_ax.texts if to_rgb(label.get_color()) in endpoint_colours}
    for label in fig.findobj(Text):
        if label not in output_symbols:
            label.set_color("black")
    for ax in plots:
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    fig.savefig(HERE/"ionisation_main.pdf", metadata={"CreationDate": None})
    fig.savefig(HERE/"ionisation_main.png", dpi=300)
    fig.savefig(HERE/"ionisation_main.svg", metadata={"Date": None})
    plt.close(fig)
    (HERE/"caption.txt").write_text(CAPTION, encoding="utf-8")
    outputs = ["ionisation_main.pdf", "ionisation_main.png", "ionisation_main.svg", "caption.txt",
               "panel_a_counts.csv", "panel_b_estimates.csv", "panel_b_displayed_estimates.csv",
               "panel_b_seeds.csv", "panel_b_vd_scatter_points.csv", "panel_c_estimates.csv", "panel_c_seeds.csv"]
    manifest = {
        "sources": SOURCES, "single_ph_sources": new_sources, "single_ph_metadata": new_metadata,
        "outputs": {name: sha256((HERE/name).read_bytes()).hexdigest() for name in outputs},
        "render_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "presentation": "Endpoint-coloured estimates and schematic symbols with black labels",
        "main_cohort_structures": 327, "main_cohort_groups": 238, "seeds": SEEDS,
        "panel_a_vector_source": SCHEMATIC_SOURCE.relative_to(HERE).as_posix(),
        "panel_c_endpoint": "Vd", "panel_c_category_encoding": "Borderless circles with category-specific colours",
        "panel_d_comparator": "REF0", "panel_d_x_limits_pct": limits,
        "panel_d_estimates": len(estimates), "panel_d_seed_values": len(paired_seeds),
        "legacy_plot_data_filenames": {"panel_b_vd_scatter_points.csv": "C",
                                       "panel_c_estimates.csv": "D", "panel_c_seeds.csv": "D"},
    }
    (HERE/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Created", HERE/"ionisation_main.pdf")


if __name__ == "__main__":
    main()
