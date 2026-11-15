import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GINConv, GraphConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

TASKS = ["t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax"]
TASKS_WITH_UNITS = ["t_half (h)", "Vd (L)", "CL (L/h)", "F (%)", "PPB (%)",
                    "Cmax (uM)", "Tmax (h)"]

#: Message-passing operators selectable via ``backbone``.
BACKBONES = ("GraphConv", "GIN", "GAT", "AttentiveFP")

# Feature widths produced by pipknet.featurizers. GraphConv and GAT can infer the
# input width lazily (``-1``), but GIN's MLP and AttentiveFP need it concretely.
# Verified against featurizers.atom_to_feature_vector: 14 atom types + 5 degrees
# + 3 formal charges + 1 aromatic flag = 23; bonds are 4 types + conjugated +
# in_ring = 6.
ATOM_FEATURE_DIM = 23
BOND_FEATURE_DIM = 6

#: GAT attention heads. With ``concat=True`` each head emits ``hidden_dim //
#: GAT_HEADS`` channels so the post-convolution width stays ``hidden_dim``.
GAT_HEADS = 4
#: AttentiveFP readout iterations (its ``num_timesteps``), fixed across runs.
ATTENTIVEFP_TIMESTEPS = 2


class GNNMultitask(nn.Module):
    """Multi-task graph regressor for the seven PK endpoints.

    ``n_layers`` message-passing layers + global mean pooling produce a
    structural embedding; when ``use_ion`` is set, a learnable ionisation-state
    embedding is concatenated before a two-layer MLP head. The physiology-informed
    mass-balance constraint is applied in the loss (see ``engine.py``), not in
    the architecture, so no extra parameters are needed here.

    The default ``backbone="GraphConv", n_layers=2`` is the published PIPK-Net
    architecture; the other backbones exist for the depth/operator ablation.
    ``AttentiveFP`` is a complete model with its own attentive readout, so it
    replaces both the convolution stack and the mean pooling; it is the only
    backbone that consumes ``edge_attr``.

    Args:
        hidden_dim: width of the message-passing layers and MLP hidden layer.
        ion_dim: number of ionisation classes (anionic/cationic/neutral/zwitterionic).
        ion_emb_dim: dimensionality of the learnable ionisation embedding.
        out_dim: number of regression targets (7).
        use_ion: whether to concatenate the ionisation embedding (variants B/C).
        dropout: dropout probability applied after pooling and the first MLP layer.
        backbone: message-passing operator, one of :data:`BACKBONES`.
        n_layers: number of message-passing layers (AttentiveFP: ``num_layers``).
    """

    def __init__(self, hidden_dim: int = 128, ion_dim: int = 4, ion_emb_dim: int = 16,
                 out_dim: int = len(TASKS), use_ion: bool = True, dropout: float = 0.1,
                 backbone: str = "GraphConv", n_layers: int = 2):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.use_ion = use_ion
        self.backbone = backbone
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        if backbone == "AttentiveFP":
            # Complete model including readout: out_channels is the graph embedding
            # handed to the ion-concat + MLP head, so it must equal hidden_dim.
            self.convs = None
            self.attentive_fp = AttentiveFP(
                in_channels=ATOM_FEATURE_DIM, hidden_channels=hidden_dim,
                out_channels=hidden_dim, edge_dim=BOND_FEATURE_DIM,
                num_layers=n_layers, num_timesteps=ATTENTIVEFP_TIMESTEPS,
                dropout=dropout,
            )
        else:
            self.attentive_fp = None
            self.convs = nn.ModuleList(
                self._make_conv(backbone, li, hidden_dim) for li in range(n_layers)
            )

        self.ion_embedding = nn.Embedding(ion_dim, ion_emb_dim) if use_ion else None
        fc_in = hidden_dim + (ion_emb_dim if use_ion else 0)

        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self._register_load_state_dict_pre_hook(_remap_legacy_conv_keys)

    @staticmethod
    def _make_conv(backbone: str, layer_idx: int, hidden_dim: int) -> nn.Module:
        """One message-passing layer; layer 0 consumes raw atom features."""
        first = layer_idx == 0
        if backbone == "GraphConv":
            # -1 keeps the published lazy-init construction byte-for-byte.
            return GraphConv(-1 if first else hidden_dim, hidden_dim)
        if backbone == "GIN":
            # GINConv has no lazy init: its MLP needs a concrete input width.
            in_dim = ATOM_FEATURE_DIM if first else hidden_dim
            return GINConv(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        if backbone == "GAT":
            # concat=True emits out_channels * heads, so divide to preserve width.
            if hidden_dim % GAT_HEADS != 0:
                raise ValueError(f"backbone='GAT' requires hidden_dim divisible by "
                                 f"heads={GAT_HEADS}; got hidden_dim={hidden_dim}")
            in_dim = ATOM_FEATURE_DIM if first else hidden_dim
            return GATConv(in_dim, hidden_dim // GAT_HEADS, heads=GAT_HEADS, concat=True)
        raise ValueError(f"unhandled backbone {backbone!r}")

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.attentive_fp is not None:
            # AttentiveFP performs its own attentive readout -> already (B, hidden).
            x = self.attentive_fp(x, edge_index, data.edge_attr, batch)
        else:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = global_mean_pool(x, batch)
        x = self.dropout(x)

        if self.use_ion and hasattr(data, "ion_feat"):
            ion = data.ion_feat
            if ion.dim()==2: 
                ion = ion.view(-1)
            x = torch.cat([x, self.ion_embedding(ion)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 7)


def _remap_legacy_conv_keys(state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    """Load pre-ModuleList checkpoints, which stored ``conv1.``/``conv2.``.

    Every published A/B/C/D checkpoint predates the configurable backbone and
    names its two GraphConv layers ``conv1``/``conv2``. Callers load with
    ``strict=False``, so without this remap those weights would be silently
    dropped and the model would run with randomly initialised convolutions.
    """
    for key in list(state_dict.keys()):
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        head, _, rest = suffix.partition(".")
        if head.startswith("conv") and head[4:].isdigit():
            layer_idx = int(head[4:]) - 1          # conv1 -> convs.0
            state_dict[f"{prefix}convs.{layer_idx}.{rest}"] = state_dict.pop(key)
