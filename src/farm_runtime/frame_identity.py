"""Bridge registered RGBD clocks to independent capture groups.

timestamp_ns is a synthetic processing clock, not elapsed video time.
physical_timestamp is the legacy geometry name for a capture-group key.
Explicit legacy keys may differ from the clock and are preserved by group.
"""

from __future__ import annotations


def normalize_capture_timestamps(frames: list[dict]) -> list[dict]:
    """Return copies with capture keys; reject contradictory camera grouping."""
    clocks: dict[str, int] = {}
    explicit: dict[str, str] = {}
    for row in frames:
        identity = str(row.get("frame_id", "")).strip()
        clock = row.get("timestamp_ns")
        if not identity or type(clock) is not int or clock <= 0:
            raise ValueError("registered frames need frame_id and positive integer timestamp_ns")
        if identity in clocks and clocks[identity] != clock:
            raise ValueError("one timestamp_ns per capture group required")
        clocks[identity] = clock
        value = row.get("physical_timestamp")
        key = "" if value is None else str(value).strip()
        if key:
            if not key.isascii() or not key.isdecimal() or int(key) <= 0:
                raise ValueError("physical_timestamp must be a positive integer capture key")
            key = str(int(key))
            if identity in explicit and explicit[identity] != key:
                raise ValueError("conflicting physical_timestamp keys within capture group")
            explicit[identity] = key
    if len(set(clocks.values())) != len(clocks):
        raise ValueError("timestamp_ns collides across different capture groups")
    keys = {identity: explicit.get(identity, str(clock)) for identity, clock in clocks.items()}
    if len(set(keys.values())) != len(keys):
        raise ValueError("physical_timestamp collides across different capture groups")
    return [dict(row, physical_timestamp=keys[str(row["frame_id"]).strip()]) for row in frames]
