"""The protocol document must agree with the header the firmware compiles.

Written after PROTOCOL.md spent three days telling implementers that
ACTIVATION_STATE bytes 10-11 were reserved and to write zero, while the
firmware was putting the live EMA baseline there. A document that is merely
stale is a nuisance; one that instructs a writer to overwrite a field in use is
a defect, and nothing failed -- it was found by re-reading.

These checks are deliberately shallow. They compare the two things that can be
compared without a second parser for C: the stated payload lengths, and whether
each field table actually tiles its payload without a gap or an overlap. Field
*meaning* still has to be kept honest by hand.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "PROTOCOL.md"
HEADER = ROOT / "src" / "emg_packet.h"

# Section heading -> the #define that fixes that payload's size. RAW is absent
# because its payload is variable by design.
SIZED_SECTIONS = {
    "INFO": "EMG_INFO_PAYLOAD_SIZE",
    "INTENT": "EMG_INTENT_PAYLOAD_SIZE",
    "ACTIVATION_STATE": "EMG_ACTIVATION_STATE_PAYLOAD_SIZE",
    "SET_ACTIVATION": "EMG_SET_ACTIVATION_PAYLOAD_SIZE",
}

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|")


def header_defines():
    text = HEADER.read_text(encoding="utf-8")
    found = {}
    for match in re.finditer(r"^#define\s+(EMG_\w+)\s+(\d+)u?\s*$", text,
                             re.MULTILINE):
        found[match.group(1)] = int(match.group(2))
    return found


def document_sections():
    """Each `### 0xNN NAME` section's field rows and stated payload length."""
    text = DOCUMENT.read_text(encoding="utf-8")
    sections = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^### `0x[0-9a-fA-F]+` (\w+)", line)
        if heading:
            current = heading.group(1)
            sections[current] = {"rows": [], "stated": None}
            continue
        if current is None:
            continue
        row = ROW.match(line)
        if row:
            sections[current]["rows"].append(
                (int(row.group(1)), int(row.group(2)), row.group(3))
            )
        stated = re.match(r"^Payload length (\d+)\.", line)
        if stated:
            sections[current]["stated"] = int(stated.group(1))
    return sections


@pytest.mark.parametrize("name,define", sorted(SIZED_SECTIONS.items()))
def test_the_documented_payload_length_is_the_compiled_one(name, define):
    stated = document_sections()[name]["stated"]
    compiled = header_defines()[define]
    assert stated == compiled, (
        f"PROTOCOL.md says {name} is {stated} bytes, {define} says {compiled}"
    )


@pytest.mark.parametrize("name", sorted(SIZED_SECTIONS))
def test_the_field_table_tiles_the_payload_exactly(name):
    """Every byte accounted for once: no gap to write into, no overlap.

    This is the check that would have caught the reserved-bytes defect only if
    the field had been removed rather than renamed, so it is not sufficient on
    its own -- but a table that no longer tiles is always wrong, and that is
    worth failing on.
    """
    section = document_sections()[name]
    offset = 0
    for start, size, field in section["rows"]:
        assert start == offset, (
            f"{name}: `{field}` starts at {start}, expected {offset}"
        )
        offset += size
    assert offset == section["stated"], (
        f"{name}: fields cover {offset} bytes, stated length is "
        f"{section['stated']}"
    )


def test_the_documented_frame_sizes_are_the_compiled_ones():
    defines = header_defines()
    text = DOCUMENT.read_text(encoding="utf-8")
    header_rows = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*`(?:magic|"
                             r"version|type|length|sequence|timestamp_us)`",
                             text, re.MULTILINE)
    ]
    assert header_rows, "the frame-header table was not found"
    assert sum(size for _, size in header_rows) == defines["EMG_HEADER_SIZE"]
    assert f"`{defines['EMG_PROTOCOL_VERSION']}` for this document" in text
