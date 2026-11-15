import pytest
import torch

from prepare_mass_balance_features import molecular_graph, pool_final_hidden, tokenize_structures


def test_pooling_uses_only_attended_non_special_positions():
    hidden = torch.tensor([
        [[100.0, 100.0], [2.0, 4.0], [4.0, 10.0], [100.0, 100.0], [float("nan"), 100.0]],
        [[100.0, 100.0], [8.0, 11.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
    ])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    special = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 1, 0, 0]])

    pooled = pool_final_hidden(hidden, attention, special)
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 7.0], [8.0, 11.0]]))


def test_pooling_rejects_no_content_tokens():
    with pytest.raises(ValueError, match="no content tokens"):
        pool_final_hidden(torch.ones(1, 2, 1), torch.ones(1, 2), torch.ones(1, 2))


def test_token_overflow_raises_with_structure_id():
    class Tokenizer:
        all_special_ids = [0, 1, 2]

        def __call__(self, smiles, **options):
            assert smiles == ["CCCC"]
            assert options["add_special_tokens"] is True
            assert options["truncation"] is False
            return {"input_ids": [[0, 20, 20, 20, 20, 2]], "special_tokens_mask": [[1, 0, 0, 0, 0, 1]]}

    with pytest.raises(ValueError, match=r"limit 5 exceeded.*too-long.*6"):
        tokenize_structures(Tokenizer(), ["too-long"], ["CCCC"], limit=5)
    _, lengths = tokenize_structures(Tokenizer(), ["exact-limit"], ["CCCC"], limit=6)
    assert lengths == [6]


def test_graph_keeps_disconnected_fragments_and_supports_empty_edges():
    graph = molecular_graph("ethanol-chloride", "CCO.[Cl-]")
    assert set(graph.keys()) == {"x", "edge_index", "edge_attr"}
    assert graph.x.shape == (4, 23)
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 6)
    assert 3 not in graph.edge_index

    ions = molecular_graph("sodium-chloride", "[Na+].[Cl-]")
    assert ions.x.shape == (2, 23)
    assert ions.edge_index.shape == (2, 0)
    assert ions.edge_attr.shape == (0, 6)
