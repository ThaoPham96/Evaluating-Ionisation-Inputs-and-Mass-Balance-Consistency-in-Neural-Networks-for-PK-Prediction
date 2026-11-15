"""Shared, vector-only mass-balance training schematic for Figure 1A."""
from pathlib import Path
import sys
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import rdDepictor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import figstyle as fs
SMALL, INK = fs.SMALL, fs.INK

def molecule(ax, centre, width):
    """Paracetamol as an illustrative structure; RDKit coordinates drawn as vector bonds."""
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(mol)
    xy = np.array(mol.GetConformer().GetPositions())[:, :2]
    xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
    xy *= width / np.ptp(xy[:, 0])
    xy += centre
    for bond in mol.GetBonds():
        a, b = xy[bond.GetBeginAtomIdx()], xy[bond.GetEndAtomIdx()]
        v = b - a
        normal = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        offsets = (-.17, .17) if bond.GetBondTypeAsDouble() == 2 else (0,)
        for off in offsets:
            aa, bb = a + off * normal, b + off * normal
            ax.plot([aa[0], bb[0]], [aa[1], bb[1]], color=INK, lw=.9, zorder=2, solid_capstyle="round")
    for atom, (x, y) in zip(mol.GetAtoms(), xy):
        if atom.GetSymbol() != "C":
            ax.text(x, y, atom.GetSymbol() + ("H" if atom.GetTotalNumHs() else ""), ha="center", va="center",
                    fontsize=SMALL, color=INK,
                    bbox=dict(fc="white", ec="none", pad=.3), zorder=5)

def arrow(ax, a, b, color, dashed=False, scale=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=scale,
                                lw=1.0, color=color, shrinkA=0, shrinkB=0,
                                linestyle=(0, (3, 2)) if dashed else '-', zorder=4))


def route(ax, points, color, dashed=False):
    xy = np.asarray(points)
    if len(points) > 2:
        ax.plot(xy[:-1, 0], xy[:-1, 1], color=color, lw=1.0,
                ls=(0, (3, 2)) if dashed else '-', solid_capstyle='round', zorder=3)
    arrow(ax, points[-2], points[-1], color, dashed)


def rounded(ax, xy, width, height, edge, fill='white', rounding=.5, lw=.8):
    ax.add_patch(FancyBboxPatch(xy, width, height,
                 boxstyle=f'round,pad=0,rounding_size={rounding}',
                 facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))

def draw_schematic(fig, box, letter_position, letter_size=10):
    ax = fig.add_axes(box)
    width, height = fig.get_size_inches()
    ymax = 100 * (height * box[3]) / (width * box[2])
    ax.set(xlim=(0, 100), ylim=(0, ymax), aspect='equal')
    ax.axis('off')
    ink = '#263541'
    fig.text(*letter_position, 'A', va='top', fontsize=letter_size,
             fontweight='bold', color='black')
    mid = 24.0
    for x, label in [(11, 'Molecular structure'), (40, 'Neural network'),
                     (77.5, 'Predicted PK endpoints')]:
        ax.text(x, 32.6, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='black')
    molecule(ax, np.array([11., mid]), 16.5)
    arrow(ax, (21.5, mid), (27.0, mid), ink)
    layers = [(31, 3), (37, 4), (43, 4), (49, 3)]
    nodes = [(x, mid + 3.0 * (np.arange(n) - (n - 1) / 2)) for x, n in layers]
    for (x1, ys1), (x2, ys2) in zip(nodes[:-1], nodes[1:]):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color='#CBD7DF', lw=.55, zorder=0)
    for x, ys in nodes:
        for y in ys:
            ax.add_patch(Circle((x, y), .85, fc=fs.BOX, ec=fs.AXES, lw=.8, zorder=3))
    arrow(ax, (52.5, mid), (57.6, mid), ink)

    # Outer group includes all seven supervised outputs. The orange inner group
    # selects exactly the three quantities used by the mass-balance penalty.
    rounded(ax, (58.6, 16.8), 37.4, 13.2, ink, lw=.8)
    rounded(ax, (59.7, 21.0), 35.2, 7.9, fs.LOSS, fs.LOSS_FILL, lw=.8)
    for i, key in enumerate(['t_half', 'Vd', 'CL']):
        x = 65.5 + i * 11.7
        ax.text(x, 26.1, fs.ENDPOINTS[key][1], ha='center', va='center',
                fontsize=12, color=fs.ENDPOINT_COLOURS[key])
        label = {'t_half':'Half-life', 'Vd':'Volume', 'CL':'Clearance'}[key]
        ax.text(x, 23.1, label, ha='center', va='center', fontsize=7, color='black')
    for i, key in enumerate(['Cmax', 'Tmax', 'F', 'PPB']):
        ax.text(63.2 + i * 9.4, 18.7, fs.ENDPOINTS[key][1], ha='center',
                va='center', fontsize=9.5, color=fs.ENDPOINT_COLOURS[key])

    # Aligned loss boxes. L_MSE is the supervised term called L_sup in Methods.
    x0, x1 = 35., 94.
    rounded(ax, (x0, 10.4), x1-x0, 4.6, ink)
    ax.text((x0+x1)/2, 12.7, r'Supervised MSE loss ($L_{\mathrm{MSE}}$)',
            ha='center', va='center', fontsize=9.2, color='black')
    rounded(ax, (x0, 1.1), x1-x0, 7.3, fs.LOSS, fs.LOSS_FILL)
    ax.text((x0+x1)/2, 6.4, 'Mass-balance training loss', ha='center',
            va='center', fontsize=9, fontweight='bold', color='black')
    equation_parts = [
        (r'$\lambda\,\mathrm{mean}[(\log_{10}$', 'black'),
        (fs.ENDPOINTS['t_half'][1], fs.ENDPOINT_COLOURS['t_half']),
        (r'$-\log_{10}$', 'black'),
        (fs.ENDPOINTS['Vd'][1], fs.ENDPOINT_COLOURS['Vd']),
        (r'$+\log_{10}$', 'black'),
        (fs.ENDPOINTS['CL'][1], fs.ENDPOINT_COLOURS['CL']),
        (r'$-\log_{10}(\ln 2))^2]$', 'black'),
    ]
    equation = HPacker(children=[TextArea(part, textprops={'fontsize':8, 'color':c})
                                 for part,c in equation_parts], align='baseline', pad=0, sep=.8)
    ax.add_artist(AnnotationBbox(equation, ((x0+x1)/2, 3.55), xycoords='data',
                                 frameon=False, pad=0))

    # Separate all-output and triad routes enter the middle of the correct box.
    route(ax, [(96., 18.7), (97.3, 18.7), (97.3, 12.7), (94.25, 12.7)], ink)
    route(ax, [(95.0, 25.0), (99.1, 25.0), (99.1, 4.75), (94.25, 4.75)], fs.LOSS)

    # Summing node combines both loss terms before a single training-feedback path.
    joint = (28.8, 8.7)
    route(ax, [(34.8, 12.7), (28.8, 12.7), (28.8, 9.85)], ink)
    route(ax, [(34.8, 4.75), (28.8, 4.75), (28.8, 7.55)], fs.LOSS)
    ax.add_patch(Circle(joint, 1.05, fc='white', ec=ink, lw=.9, zorder=5))
    ax.text(*joint, '+', ha='center', va='center', color='black', fontsize=11, zorder=6)
    route(ax, [(27.7, 8.7), (23.9, 8.7), (23.9, 21.0), (29.7, 21.0)], ink, dashed=True)

    return ax
