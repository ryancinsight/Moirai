#!/usr/bin/env python3
"""Compare three completed real keyboard cells without inferring delivery absence.

The caller admits all three native process exits in one explicitly selected
observer group before invocation. This reader
checks their retained values; it does not run input, a browser, or compiled code.
For (Default, EnableS, SkipS), reports (0,0,1) support a bounded property-specific
difference, (0,1,1) a setter-path difference, and (1,1,1) no reproduction. Other
patterns are inconclusive. No pattern establishes a Metis failure cause.
"""

import argparse
import hashlib
import json
from pathlib import Path
import stat


PAYLOAD_NAMES = ("bootstrap.js", "favicon.svg", "index.html",
                 "webview_keyboard_fixture.js", "webview_keyboard_fixture_bg.wasm")
LETTERS = (68, 83, 80)
FLAGS = ("ctrl", "alt", "shift", "meta", "repeat", "trusted")
MODIFIERS = (False, True, True, False, False, True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def closed(value, fields):
    require(type(value) is dict and set(value) == set(fields.split()),
            f"closed fields differ: {fields}")


def integer(value, maximum, minimum=0):
    require(type(value) is int and minimum <= value <= maximum, "integer domain differs")
    return value


def text(value):
    require(type(value) is str and value and "\0" not in value, "nonempty native text required")
    value.encode("utf-8", errors="strict")
    return value


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    path = path.resolve(strict=True)
    require(stat.S_ISREG(path.stat().st_mode), "actual evidence must be a regular file")
    raw = path.read_bytes()
    return loads(raw), {"path": str(path), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}


def payload_identity(path):
    path = path.resolve(strict=True)
    require(path.is_dir(), "actual compiled payload directory required")
    require(sorted(p.name for p in path.iterdir()) == list(PAYLOAD_NAMES),
            "actual payload must contain the five admitted members")
    files = []
    for name in PAYLOAD_NAMES:
        member = path / name
        require(stat.S_ISREG(member.lstat().st_mode), "payload members must be regular files")
        with member.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files.append({"path": name, "bytes": member.stat().st_size, "sha256": digest})
    builder, receipt = read_json(path.parent / "evidence.json")
    require(type(builder) is dict and type(builder.get("schema")) is int and builder["schema"] == 1
            and builder.get("kind") == "actual-rust-native-keyboard-payload",
            "actual keyboard builder receipt required")
    require(Path(text(builder["payload"])).samefile(path), "builder payload identity differs")
    require(builder["files"] == files, "builder hashes differ from actual payload bytes")
    return {"directory": str(path), "files": files, "builder_receipt": receipt,
            "provenance": "Present bytes match the retained builder report; this comparison does not compile them."}


def notice(value):
    require(type(value) is dict, "document notice must be an object")
    kind = value.get("kind")
    fields = {"ready": "kind", "key": "kind sequence keyboard",
              "closed": "kind key_events publication_error bridge_retained dom_active dom_retained"}
    require(kind in fields, "unknown document notice")
    closed(value, fields[kind])
    if kind == "key":
        integer(value["sequence"], 2**32 - 1, 1)
        keyboard = value["keyboard"]
        closed(keyboard, "key code ctrl alt shift meta repeat trusted")
        for key in ("key", "code"):
            require(type(keyboard[key]) is str, "keyboard key/code must retain complete strings")
            keyboard[key].encode("utf-8", errors="strict")
        require(all(type(keyboard[flag]) is bool for flag in FLAGS), "keyboard flag type differs")
    if kind == "closed":
        integer(value["key_events"], 2**32 - 1)
        require(value["publication_error"] is None, "document publication fails")
        for key in ("bridge_retained", "dom_active", "dom_retained"):
            require(type(value[key]) is int and value[key] == 0, "document owner remains retained")
    return kind


def document(value):
    closed(value, "rows originals original_bytes")
    rows, originals = value["rows"], value["originals"]
    require(type(rows) is list and type(originals) is list
            and 2 <= len(rows) <= 66 and len(rows) == len(originals), "document count differs")
    require(all(type(original) is str for original in originals), "raw SDK notices must be strings")
    sizes = [len(original.encode("utf-8", errors="strict")) for original in originals]
    require(sum(sizes) <= 65_536, "actual SDK original bytes exceed aggregate cap")
    require(type(value["original_bytes"]) is int and value["original_bytes"] == sum(sizes),
            "actual SDK original-byte counter differs")
    require([loads(original) for original in originals] == rows, "typed notices differ from raw SDK JSON")
    kinds = [notice(row) for row in rows]
    require(kinds == ["ready", *(["key"] * (len(rows) - 2)), "closed"], "document lifecycle order differs")
    keys = rows[1:-1]
    require([row["sequence"] for row in keys] == list(range(1, len(keys) + 1)),
            "document key sequence is not contiguous")
    require(rows[-1]["key_events"] == len(keys), "Closed key count differs")
    selected = {}
    for letter in LETTERS:
        code = "Key" + chr(letter)
        actual = [row for row in keys if row["keyboard"]["code"] == code]
        require(len(actual) <= 1, "one physical chord produces duplicate letter notices")
        require(all(tuple(row["keyboard"][flag] for flag in FLAGS) == MODIFIERS for row in actual),
                "document control flags/trust differ")
        require(letter == 83 or len(actual) == 1, "D/P document controls are not observed")
        selected[code] = actual
    return {"key_events": len(keys), "original_bytes": sum(sizes), "controls": selected,
            "closed": rows[-1], "raw_notice_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in originals]}


def chord(value, letter):
    closed(value, "letter focus foreground thread process layout strokes admitted insertion_error releases")
    require(type(value["letter"]) is int and value["letter"] == letter, "D/S/P chord order differs")
    for name in ("focus", "foreground", "layout"):
        text(value[name])
    require(value["focus"] != value["foreground"], "native receipt lacks descendant focus")
    for name in ("thread", "process"):
        integer(value[name], 2**32 - 1, 1)
    require(type(value["admitted"]) is int and value["admitted"] == 6
            and value["insertion_error"] is None and value["releases"] == [], "physical insertion differs")
    strokes = value["strokes"]
    require(type(strokes) is list and len(strokes) == 6, "six actual strokes required")
    require([row["key"] for row in strokes] == [18, 16, letter, letter, 16, 18], "physical key order differs")
    for index, row in enumerate(strokes):
        closed(row, "key scan flags")
        integer(row["key"], 2**32 - 1)
        integer(row["scan"], 255, 1)
        flags = integer(row["flags"], 2**32 - 1)
        require(flags & ~11 == 0 and flags & 8 and bool(flags & 2) == (index >= 3),
                "physical scan/release flags differ")
    for down, up in zip(strokes[:3], reversed(strokes[3:])):
        require(down["key"] == up["key"] and down["scan"] == up["scan"]
                and down["flags"] | 2 == up["flags"], "release does not reuse actual mapping")
    return {"letter": letter, "layout": value["layout"], "strokes": strokes,
            "recipient": {key: value[key] for key in ("focus", "foreground", "thread", "process")}}


def attempt(value):
    closed(value, "kind key lparam held handled_before browser_before setter_succeeded browser_after handled_after failure")
    integer(value["kind"], 3)
    integer(value["key"], 2**32 - 1)
    integer(value["lparam"], 2**31 - 1, -(2**31))
    require(type(value["held"]) is list and len(value["held"]) == 5, "native modifier domain differs")
    for item in value["held"]:
        integer(item, 2**15 - 1, -(2**15))
    for key in ("handled_before", "browser_before", "browser_after", "handled_after"):
        require(type(value[key]) is bool, "actual native getter value is missing")
    require(value["setter_succeeded"] is None or type(value["setter_succeeded"]) is bool,
            "setter result type differs")
    require(value["failure"] is None, "native callback reports a getter/setter failure")
    require(value["handled_before"] == value["handled_after"], "Handled changes")
    require(bool((value["lparam"] & 0xffffffff) & (1 << 31)) == (value["kind"] in (1, 3)),
            "native kind/release metadata differ")


def fresh_letter(value, letter):
    held = value["held"]
    return (value["key"] == letter and value["kind"] in (0, 2)
            and not value["lparam"] & (1 << 30)
            and held[0] >= 0 and held[1] < 0 and held[2] < 0 and held[3] >= 0 and held[4] >= 0)


def selected_observer(value, observer):
    require(type(observer) is str and observer in ("selected", "none"),
            "explicit selected/none observer group required")
    require(type(value) is bool and value is (observer == "selected"),
            "actual observer selection differs")
    return value


def cell(path, policy, payload, observer):
    value, identity = read_json(path)
    closed(value, "schema policy observer_selected runtime runtime_error chords controller document operation_error callback_cleanup host_cleanup destroyed profile_cleanup coverage payload_directory payload_files")
    require(type(value["schema"]) is int and value["schema"] == 2 and value["policy"] == policy,
            "native cell schema/policy differs")
    selected = selected_observer(value["observer_selected"], observer)
    text(value["runtime"])
    for key in ("runtime_error", "operation_error", "callback_cleanup", "host_cleanup", "profile_cleanup"):
        require(value[key] is None, f"native {key} is retained")
    require(type(value["destroyed"]) is int and value["destroyed"] == 1, "native destruction differs")
    text(value["coverage"])
    original = Path(text(value["payload_directory"]))
    require(original.is_absolute() and original.samefile(payload["directory"]), "cell payload identity differs")
    expected = {row["path"]: row["bytes"] for row in payload["files"]}
    require(type(value["payload_files"]) is dict and value["payload_files"] == expected
            and all(type(size) is int for size in value["payload_files"].values()), "cell payload sizes differ")
    require(type(value["chords"]) is list and len(value["chords"]) == 3, "three physical chords required")
    chords = [chord(row, letter) for row, letter in zip(value["chords"], LETTERS)]
    records = value["controller"]
    require(type(records) is list and 1 <= len(records) <= 64, "controller bounded record count differs")
    for row in records:
        attempt(row)
    matching = [row for row in records if fresh_letter(row, 83)]
    require(len(matching) == 1, "one actual fresh S-down intervention domain required")
    target = matching[0]
    require(target["handled_before"] is False and target["handled_after"] is False
            and target["browser_before"] is True, "S entry getters differ")
    setters = [row for row in records if row["setter_succeeded"] is not None]
    require((not setters) if policy == "Default" else (len(setters) == 1 and setters[0] is target
            and target["setter_succeeded"] is True), "setter ownership differs")
    require(target["browser_after"] is (policy != "SkipS"), "selected browser property result differs")
    for row in records:
        require(row is target or (row["setter_succeeded"] is None
                and row["browser_before"] == row["browser_after"]), "other callback property changes")
    for mapped in chords:
        rows = [row for row in records if fresh_letter(row, mapped["letter"])]
        require(len(rows) == 1, "one actual native D/S/P-down control required")
        bits = rows[0]["lparam"] & 0xffffffff
        stroke = mapped["strokes"][2]
        require((bits >> 16) & 255 == stroke["scan"]
                and bool(bits & (1 << 24)) == bool(stroke["flags"] & 1), "native scan differs from actual input")
    return {"policy": policy, "observer_selected": selected,
            "receipt": identity, "runtime": value["runtime"],
            "chords": chords, "controller_records": len(records), "selected_S_callback": target,
            "document": document(value["document"]), "declared_coverage": value["coverage"],
            "payload_directory": value["payload_directory"], "payload_files": value["payload_files"]}


def classify(pattern):
    require(type(pattern) is tuple and len(pattern) == 3 and all(type(v) is bool for v in pattern),
            "three report-presence Booleans required")
    return {(False, False, True): "BOUNDED_PROPERTY_SPECIFIC_DIFFERENCE",
            (False, True, True): "BOUNDED_SETTER_PATH_DIFFERENCE",
            (True, True, True): "NOT_REPRODUCED"}.get(pattern, "INCONCLUSIVE")


def compare(default, enabled, skipped, payload_path, observer):
    payload = payload_identity(payload_path)
    cells = [cell(path, policy, payload, observer) for path, policy in
             zip((default, enabled, skipped), ("Default", "EnableS", "SkipS"))]
    require(len({row["runtime"] for row in cells}) == 1, "actual runtime versions differ")
    vectors = [[{key: row[key] for key in ("letter", "layout", "strokes")} for row in item["chords"]]
               for item in cells]
    require(vectors[0] == vectors[1] == vectors[2], "actual layouts/stroke vectors differ")
    pattern = tuple(bool(row["document"]["controls"]["KeyS"]) for row in cells)
    require(pattern[2], "SkipS actual document S is required by the native gate")
    return {"schema": 2, "observer": observer,
            "classification": classify(pattern), "S_notice_pattern": list(pattern),
            "cells": cells, "payload": payload, "limits": [
                "The caller admits all three real native process exits; this reader does not rerun them.",
                "This comparison covers only its explicit observer group; another group's result cannot satisfy it.",
                "Copied payload identity is an executed native-source byte assertion; copied hashes are not independently archived here.",
                "The same native filesystem and still-existing original payload paths are required; no path rewriting occurs.",
                "Absent document notices mean only unobserved in these bounded prefixes, never delivery absence.",
                "Setter/property differences do not establish a Metis root cause, renderer-wide behavior or complete keyboard/IPC coverage.",
                "No private clocks are aligned and no ordered callbacks prove renderer consumption."]}


def publish(result, path):
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as destination:
        destination.write(encoded)
    return {"output": str(path.resolve()), "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(), "classification": result["classification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("default", "enable-s", "skip-s", "payload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--observer", choices=("selected", "none"), required=True)
    args = parser.parse_args()
    result = compare(args.default, args.enable_s, args.skip_s, args.payload, args.observer)
    print(json.dumps(publish(result, args.output)))


if __name__ == "__main__":
    main()
