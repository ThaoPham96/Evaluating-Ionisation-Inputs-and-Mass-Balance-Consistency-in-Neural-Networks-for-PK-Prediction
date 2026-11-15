"""Width-matched single-pH categorical features from the frozen pH ledger."""
from __future__ import annotations

import numpy as np

from ph_ionisation_features import AUDITED_CLASSES, CATEGORIES, PH, threshold_codes


ARM_PH = {'PH2': '2.0', 'PH5': '5.0', 'PH65': '6.5'}
WIDTHS = dict.fromkeys(ARM_PH, len(CATEGORIES))


def feature_matrix(arm, code_rows):
    """Return one four-channel block in the same category order as REF74."""
    column = PH.index(ARM_PH[arm])
    codes = np.asarray(code_rows)
    if (codes.ndim != 2 or codes.shape[1] != len(PH) or codes.dtype.kind not in 'iu' or
            np.any((codes < 0) | (codes >= len(CATEGORIES)))):
        raise ValueError('Expected four integer pH codes per structure')
    return np.eye(len(CATEGORIES), dtype=np.float32)[codes[:, column]]


def validate_ledger_codes(records, codes):
    """Verify frozen pKa threshold derivation and identity-matched pH 7.4 labels."""
    for record, entry in zip(records, codes, strict=True):
        expected = [block[2] for block in threshold_codes(record['pka_acid'], record['pka_base'])]
        if list(entry) != expected or AUDITED_CLASSES[expected[0]] != record['ionisation_class']:
            raise ValueError(f"Categorical pH ledger disagrees with pKa data for {record['structure_id']}")
