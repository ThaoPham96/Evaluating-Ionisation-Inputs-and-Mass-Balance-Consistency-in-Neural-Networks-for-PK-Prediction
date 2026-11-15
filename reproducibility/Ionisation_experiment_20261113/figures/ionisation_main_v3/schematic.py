"""Vector reconstruction of the ionisation-only input and prediction flow."""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from ionisation_style import AXES, BLUE, BLUE_FILL, CATEGORIES, GRID, INK, MUTED, NEUTRAL_BOX, ENDPOINT_COLORS


def arrow(ax, start, end, colour=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                color=colour, lw=1.0, shrinkA=0, shrinkB=0, zorder=4))


def box(ax, x, y, width, height, fill=NEUTRAL_BOX, edge=GRID):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=.1,rounding_size=.6",
                                fc=fill, ec=edge, lw=.8, zorder=1))


def molecule(ax):
    """Use the illustrative molecule from the schematic with RDKit 2D coordinates."""
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(mol)
    xy = np.array(mol.GetConformer().GetPositions())[:, :2]
    xy -= (xy.min(axis=0) + xy.max(axis=0)) / 2
    xy *= 15 / np.ptp(xy[:, 0])
    xy += [10, 22]
    for bond in mol.GetBonds():
        a, b = xy[bond.GetBeginAtomIdx()], xy[bond.GetEndAtomIdx()]
        d = b - a
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        offsets = (-.16, .16) if bond.GetBondTypeAsDouble() == 2 else (0.,)
        for offset in offsets:
            points = np.array([a, b]) + offset * normal
            ax.plot(*points.T, color=INK, lw=1.0, zorder=2)
    for atom, (x, y) in zip(mol.GetAtoms(), xy):
        if atom.GetSymbol() != "C":
            label = atom.GetSymbol() + ("H" if atom.GetTotalNumHs() else "")
            ax.text(x, y, label, fontsize=8, ha="center", va="center", color=INK,
                    bbox={"fc": "white", "ec": "none", "pad": .3}, zorder=3)


def draw_schematic(ax):
    ax.set(xlim=(0, 112), ylim=(3, 29))
    ax.set_aspect("equal")
    ax.axis("off")
    headings = ((10, "Molecular structure"), (34, "Neural network"),
                (54.5, "Concatenate"), (74, "Prediction head"), (97, "PK endpoints"))
    for x, label in headings:
        ax.text(x, 27.5, label, fontsize=9, weight="bold", ha="center", va="center")
    molecule(ax)
    arrow(ax, (19.3, 22), (24.2, 22))

    layers = [(26, (19.2, 22, 24.8)), (34, (18.4, 20.8, 23.2, 25.6)),
              (42, (19.2, 22, 24.8))]
    for (left, starts), (right, ends) in zip(layers, layers[1:]):
        for start in starts:
            for end in ends:
                ax.plot([left, right], [start, end], color="#CFD8DF", lw=.6, zorder=1)
    for x, ys in layers:
        for y in ys:
            ax.add_patch(Circle((x, y), .66, fc=NEUTRAL_BOX, ec=AXES, lw=.75, zorder=3))
    arrow(ax, (44.1, 22), (52.8, 22))
    ax.add_patch(Circle((54.5, 22), 1.55, fc=NEUTRAL_BOX, ec=AXES, lw=.8, zorder=3))
    ax.text(54.5, 22, "+", fontsize=12, ha="center", va="center", color=MUTED, zorder=4)
    arrow(ax, (56.1, 22), (67.4, 22))
    box(ax, 68, 18.6, 12, 6.8)
    ax.text(74, 22, "MLP", fontsize=10, ha="center", va="center", weight="bold")
    arrow(ax, (80.4, 22), (84.2, 22))
    outputs = ((85, 22.4, "t_half", r"$t_{1/2}$"), (93.8, 22.4, "Vd", r"$V_d$"),
               (102.6, 22.4, "CL", r"$CL$"), (85, 17.8, "Cmax", r"$C_{\max}$"),
               (91.6, 17.8, "Tmax", r"$T_{\max}$"), (98.2, 17.8, "F", r"$F$"),
               (104.8, 17.8, "PPB", r"$PPB$"))
    for x, y, endpoint, symbol in outputs:
        width = 8 if y > 20 else 6
        box(ax, x, y, width, 3.8)
        ax.text(x + width / 2, y + 1.9, symbol, fontsize=9, ha="center", va="center",
                color=ENDPOINT_COLORS[endpoint])

    box(ax, .5, 3.5, 88.5, 11.6, fill=BLUE_FILL, edge=BLUE)
    ax.text(2.2, 13.25, "Ionisation input", color=BLUE, fontsize=9, weight="bold", va="center")
    for x, width, label in ((2.3, 19.0, r"Acidic and basic p$K_a$"), (23.5, 13.0, "Selected pH")):
        box(ax, x, 7.7, width, 3.8, fill="white")
        ax.text(x + width/2, 9.6, label, fontsize=8, ha="center", va="center")
        ax.plot([x + width/2, x + width/2], [7.6, 5.8], color=BLUE, lw=.9)
    ax.plot([11.8, 39.2, 39.2], [5.8, 5.8, 8.7], color=BLUE, lw=.9)
    arrow(ax, (39.2, 8.7), (42.0, 8.7), BLUE)

    box(ax, 42.5, 4.7, 44.9, 8.8, fill="white")
    ax.text(45, 11.7, "Four-category one-hot vector", fontsize=8, va="center", weight="bold")
    positions = ((45, 9.1), (59, 9.1), (45, 6.5), (59, 6.5))
    for (x, y), (label, colour, _) in zip(positions, CATEGORIES.values(), strict=True):
        ax.plot(x, y, marker="o", ms=3.5, mfc=colour, mec="none")
        ax.text(x + 1.15, y, label, fontsize=7.5, va="center")
    for i in range(4):
        ax.add_patch(Rectangle((77 + 1.9*i, 7.5), 1.9, 1.9, fc="white", ec=BLUE, lw=.65))
    arrow(ax, (54.5, 13.6), (54.5, 20.35), BLUE)
