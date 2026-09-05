"""Filter a native Gaussian CSR bank using an explicit final object allowlist."""

from __future__ import annotations

import numpy as np


def filter_instance_bank(bank, allowed_ids, *, source_count):
    ids = np.asarray(bank["object_ids"])
    ptr = np.asarray(bank["indptr"])
    indices = np.asarray(bank["indices"])
    allowed = sorted(set(int(i) for i in allowed_ids))
    if ids.ndim != 1 or ptr.shape != (len(ids) + 1,) or indices.ndim != 1:
        raise ValueError("invalid CSR shapes")
    if not all(np.issubdtype(x.dtype, np.integer) for x in (ids, ptr, indices)):
        raise ValueError("CSR IDs and offsets must be integers")
    if (
        ptr[0] != 0
        or ptr[-1] != len(indices)
        or np.any(np.diff(ptr) < 0)
        or np.any(np.diff(ids) <= 0)
    ):
        raise ValueError("CSR offsets or object order are invalid")
    if (
        np.any(indices < 0)
        or np.any(indices >= source_count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "Gaussian ownership must be exclusive and within the source PLY"
        )
    if set(allowed) - set(ids.tolist()):
        raise ValueError("allowlist references an object missing from the source bank")
    selection = []
    counts = []
    for i, oid in enumerate(ids):
        if int(oid) in allowed:
            selection.extend(range(int(ptr[i]), int(ptr[i + 1])))
            counts.append(int(ptr[i + 1] - ptr[i]))
    selection = np.asarray(selection, np.int64)
    result = {
        "object_ids": np.asarray(allowed, np.int32),
        "indptr": np.r_[0, np.cumsum(counts)].astype(np.int64),
        "indices": indices[selection].copy(),
    }
    for key in ("confidence", "timestamp_support"):
        values = np.asarray(bank[key])
        if values.shape != indices.shape:
            raise ValueError("CSR attributes must be row aligned")
        result[key] = values[selection].copy()
    return result
