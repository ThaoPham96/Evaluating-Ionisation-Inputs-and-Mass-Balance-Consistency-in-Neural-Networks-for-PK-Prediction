"""Editable vector schematic of an ionisation input and a training penalty."""
from __future__ import annotations

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from rdkit import Chem
from rdkit.Chem import rdDepictor

from combined_style import (AXES, BLUE, BLUE_FILL, BOX, CATEGORIES, ENDPOINT_COLORS,
                            INK, MUTED, ORANGE, ORANGE_FILL)

EXAMPLE_SMILES = 'CC(=O)Nc1ccc(O)cc1'


def arrow(ax, start, end, *, dashed=False, color=INK, lw=1.15, scale=10):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale,
                  lw=lw, color=color, linestyle=(0, (3, 2)) if dashed else '-',
                  shrinkA=0, shrinkB=0, zorder=5))


def box(ax, x, y, width, height, face=BOX, edge='#CFD8DF', radius=.6):
    patch = FancyBboxPatch((x, y), width, height,
        boxstyle=f'round,pad=.1,rounding_size={radius}', fc=face, ec=edge, lw=.8, zorder=1)
    ax.add_patch(patch)
    return patch


def molecule(ax):
    """RDKit gives connectivity/coordinates; bonds and atom text remain vectors."""
    mol = Chem.MolFromSmiles(EXAMPLE_SMILES)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(mol)
    xy = np.asarray(mol.GetConformer().GetPositions())[:, :2]
    xy -= (xy.max(axis=0) + xy.min(axis=0))/2
    xy *= 17.0 / np.ptp(xy[:, 0])
    xy += [10.2, 25.0]
    for bond in mol.GetBonds():
        a, b = xy[bond.GetBeginAtomIdx()], xy[bond.GetEndAtomIdx()]
        tangent = b-a
        normal = np.asarray([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
        for offset in (-.17, .17) if bond.GetBondTypeAsDouble() == 2 else (0.,):
            aa, bb = a+offset*normal, b+offset*normal
            ax.plot([aa[0], bb[0]], [aa[1], bb[1]], color=INK, lw=1.05, zorder=2)
    for atom, (x, y) in zip(mol.GetAtoms(), xy):
        if atom.GetSymbol() != 'C':
            label = atom.GetSymbol() + ('H' if atom.GetTotalNumHs() else '')
            ax.text(x, y, label, fontsize=7.5, ha='center', va='center',
                    color=INK, bbox={'fc':'white', 'ec':'none', 'pad':.3}, zorder=7)


def draw_schematic(ax):
    ax.set(xlim=(0, 112), ylim=(0, 33), aspect='equal')
    ax.axis('off')
    for x, label in [(10.2,'Molecular structure'), (36,'Structural encoder'),
                     (56.8,'Concatenate'), (72,'Prediction head'), (96,'Predicted PK endpoints')]:
        ax.text(x, 32, label, fontsize=8, color=INK, weight='bold', ha='center', va='center')
    molecule(ax)
    arrow(ax, (20.7,25), (25.5,25))

    layers = [(27.5,[21.5,25,28.5]), (33.5,[20.1,23.4,26.7,30]),
              (39.5,[20.1,23.4,26.7,30]), (45.5,[21.5,25,28.5])]
    for (x1, ys1),(x2,ys2) in zip(layers[:-1], layers[1:]):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1,x2],[y1,y2], color='#CFD8DF', lw=.55, zorder=1)
    for x, ys in layers:
        for y in ys:
            ax.add_patch(Circle((x,y),.72, fc=BOX, ec=AXES, lw=.7, zorder=3))
    arrow(ax,(47.5,25),(56.1,25))
    ax.add_patch(Circle((58,25),1.65, fc=BOX, ec='#BFCBD4', lw=.8,zorder=3))
    ax.text(58,25,'+',ha='center',va='center',fontsize=11,color=MUTED,zorder=6)
    arrow(ax,(59.9,25),(63,25))
    box(ax,64,21.4,12,7.2)
    ax.text(70,25,'MLP',fontsize=10,color=INK,ha='center',va='center',weight='bold',zorder=4)
    arrow(ax,(76.8,25),(81.1,25))

    for x,endpoint,symbol,name in [(82,'t_half',r'$t_{1/2}$','Half-life'),
                                   (92,'Vd',r'$V_d$','Volume'),(102,'CL',r'$CL$','Clearance')]:
        box(ax,x,23,8.3,7.2,edge=BOX)
        ax.text(x+4.15,27.3,symbol,fontsize=11,ha='center',va='center',color=ENDPOINT_COLORS[endpoint],zorder=4)
        ax.text(x+4.15,24.55,name,fontsize=7,ha='center',va='center',color=ENDPOINT_COLORS[endpoint],zorder=4)
    for x,endpoint,symbol in [(85,'Cmax',r'$C_{\max}$'),(93,'Tmax',r'$T_{\max}$'),
                              (101,'F',r'$F$'),(108,'PPB',r'$PPB$')]:
        ax.text(x,18.9,symbol,color=ENDPOINT_COLORS[endpoint],fontsize=9,ha='center',va='center')

    # A separate input branch: numerical pKas + fixed pH -> four categories -> OHE.
    box(ax,.7,1.0,52.0,14.6,face=BLUE_FILL,edge=BLUE)
    ax.text(26.7,14.05,'Ionisation input',fontsize=8.5,weight='bold',
            color=INK,ha='center',va='center',zorder=5)
    box(ax,2.5,7.2,17.0,4.9,face='white')
    ax.text(11,9.65,'Acidic + basic pKa',fontsize=7.4,ha='center',va='center',color=INK,zorder=4)
    box(ax,2.5,2.5,17.0,3.4,face='white')
    ax.text(11,4.2,'pH 7.4',fontsize=8,ha='center',va='center',color=INK,zorder=4)
    ax.plot([19.6,21.6,21.6],[9.65,9.65,7.3],color=BLUE,lw=1)
    ax.plot([19.6,21.6,21.6],[4.2,4.2,7.3],color=BLUE,lw=1)
    arrow(ax,(21.6,7.3),(24,7.3),color=BLUE,scale=8)
    box(ax,24.8,2.0,25.9,10.4,face='white')
    ax.text(37.75,10.9,'4-category one-hot input',fontsize=7.5,weight='bold',ha='center',va='center',zorder=4)
    for y,(label,color,marker) in zip([8.9,7.05,5.2,3.35], CATEGORIES):
        ax.plot(26.4,y,marker=marker,markerfacecolor=color,markeredgecolor='#333333',
                markeredgewidth=.45,markersize=3.7,zorder=4)
        ax.text(27.5,y,label,fontsize=7.1,color=INK,ha='left',va='center',zorder=4)
    # Four empty cells denote dimensions; no class is assigned to the illustrative molecule.
    for i in range(4):
        ax.add_patch(Rectangle((40.4+i*2.15,6.1),2.15,2.15,fc='white',ec=BLUE,lw=.65,zorder=4))
    ax.plot([52.8,58],[7.3,7.3],color=BLUE,lw=1.15,zorder=3)
    arrow(ax,(58,7.3),(58,23.15),color=BLUE,scale=8)

    # Dashed paths denote training, separate from the solid input/inference arrows.
    box(ax,62,1,49.1,14.6,face=ORANGE_FILL,edge=ORANGE)
    ax.text(86.55,14.05,'Mass-balance training loss',fontsize=8.5,weight='bold',
            ha='center',va='center',zorder=4)
    prefix = r'$\mathcal{L}=\mathcal{L}_{\mathrm{sup}}+\lambda\,\mathrm{mean}$'
    ax.text(86.55,10.45,prefix,fontsize=8,ha='center',va='center',color=INK,zorder=4)
    residual = [
        (r'$[(\log_{10}$', INK), (r'$t_{1/2}$', ENDPOINT_COLORS['t_half']),
        (r'$-\log_{10}$', INK), (r'$V_d$', ENDPOINT_COLORS['Vd']),
        (r'$+\log_{10}$', INK), (r'$CL$', ENDPOINT_COLORS['CL']),
        (r'$-\log_{10}(\ln 2))^2]$', INK),
    ]
    expression = HPacker(
        children=[TextArea(part, textprops={'fontsize': 8, 'color': color}) for part, color in residual],
        align='baseline', pad=0, sep=.5,
    )
    ax.add_artist(AnnotationBbox(expression, (86.55, 7.35), frameon=False, pad=0))
    ax.text(86.55,3.4,r'Applied in unstandardised $\log_{10}$ space',fontsize=7.5,color=MUTED,
            ha='center',va='center',zorder=4)
    for x in [86.15,96.15,106.15]:
        ax.plot([x,x],[22.8,21.3],ls=(0,(3,2)),color=ORANGE,lw=.85,zorder=2)
    ax.plot([86.15,111.6,111.6],[21.3,21.3,17.2],ls=(0,(3,2)),color=ORANGE,lw=.85,zorder=2)
    arrow(ax,(111.6,17.2),(108.6,15.8),dashed=True,color=ORANGE,scale=8)
    arrow(ax,(70,15.8),(70,21.15),dashed=True,color=ORANGE,scale=8)
    ax.text(71.7,18.3,'Training only',fontsize=7.1,color=INK,ha='left',va='center')

