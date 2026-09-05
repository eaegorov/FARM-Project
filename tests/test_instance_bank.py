import numpy as np
import pytest
from farm_runtime.instance_bank import filter_instance_bank


def test_final_allowlist_removes_rejected_ids_and_preserves_native_row_attributes():
    bank = dict(
        object_ids=np.array([2, 37, 185]),
        indptr=np.array([0, 2, 3, 5]),
        indices=np.array([0, 9, 2, 4, 7]),
        confidence=np.array([0.7, 0.8, 0.9, 0.6, 0.5]),
        timestamp_support=np.array([2, 3, 4, 3, 2], np.uint16),
    )
    filtered = filter_instance_bank(bank, [2, 37], source_count=10)
    assert filtered["object_ids"].tolist() == [2, 37]
    assert filtered["indices"].tolist() == [0, 9, 2]
    assert filtered["indptr"].tolist() == [0, 2, 3]
    assert filtered["timestamp_support"].tolist() == [2, 3, 4]
    empty = filter_instance_bank(bank, [], source_count=10)
    assert empty["indptr"].tolist() == [0] and len(empty["indices"]) == 0
    with pytest.raises(ValueError, match="missing"):
        filter_instance_bank(bank, [27], source_count=10)
    bank["indices"][4] = bank["indices"][0]
    with pytest.raises(ValueError, match="exclusive"):
        filter_instance_bank(bank, [2], source_count=10)
