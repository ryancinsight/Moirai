#!/usr/bin/env python3
"""Audit opt-in protocol tracing without substituting it for acceptance runs."""

import argparse
import gzip
import json
from pathlib import Path

from quic_recovery_audit import audit, diagnostics, require

COUNTS = ("congestion_blocked", "pacing_blocked", "anti_amplification_blocked",
          "data_blocked", "stream_data_blocked", "established", "pto_initial",
          "pto_handshake", "pto_data", "max_pto_backoff", "max_pto_in_flight_bytes",
          "timer_driver_wakes", "timer_lateness_ns", "max_timer_lateness_ns")
PHASES = ("authenticated", "exchange", "complete", "drain")


def protocol_trace(text, success):
    samples = []
    for line in (text or "").splitlines():
        if "protocol recovery observation" not in line:
            continue
        marker = "protocol_recovery="
        require(marker in line, "protocol recovery JSON is absent")
        sample, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
        require(sample["schema"] == 1, "unknown protocol recovery schema")
        for field in ("observed_unix_ns", "elapsed_ns"):
            require(type(sample[field]) is int and sample[field] >= 0, "invalid trace time")
        require(set(sample["counts"]) == set(COUNTS), "recovery counter inventory differs")
        require(all(type(v) is int and v >= 0 for v in sample["counts"].values()),
                "invalid recovery count")
        counts = sample["counts"]
        require(counts["max_timer_lateness_ns"] <= counts["timer_lateness_ns"],
                "timer maximum exceeds sum")
        require(counts["timer_driver_wakes"] > 0 or counts["timer_lateness_ns"] == 0,
                "lateness without a timer-driver wake")
        require(counts["established"] <= 1, "probe establishes more than one connection")
        require(sample["phase"] in ("handshake", *PHASES), "unknown probe phase")
        require(sample["observation"] in
                ("start", "sample", "success", "failure", "drained", "drain_failure"),
                "unknown observation")
        phase_names = [p["phase"] for p in sample["phases"]]
        require(phase_names == sorted(set(phase_names), key=PHASES.index), "phase order differs")
        require(sample["phase"] == (phase_names[-1] if phase_names else "handshake"),
                "phase disagrees with phase history")
        for entries in (sample["phases"], sample["timeouts"]):
            require(all(a["elapsed_ns"] <= b["elapsed_ns"] for a, b in zip(entries, entries[1:])),
                    "event clocks decrease")
            require(all(0 <= e["elapsed_ns"] <= sample["elapsed_ns"] for e in entries),
                    "event lies outside trace observation")
        for phase in sample["phases"]:
            require(set(phase["counts"]) == set(COUNTS), "phase counter inventory differs")
            require(all(type(v) is int and 0 <= v <= counts[k]
                        for k, v in phase["counts"].items()), "phase counter exceeds snapshot")
        timeouts = sample["timeouts"]
        for timeout in timeouts:
            require(timeout["phase"] in ("handshake", *PHASES), "unknown PTO phase")
            require(timeout["space"] in ("Initial", "Handshake", "Data"), "unknown PTO space")
            require(all(type(timeout[k]) is int and timeout[k] >= 0
                        for k in ("in_flight_bytes", "backoff")), "invalid PTO values")
        for space in ("Initial", "Handshake", "Data"):
            require(counts[f"pto_{space.lower()}"] == sum(t["space"] == space for t in timeouts),
                    "PTO counter differs from event inventory")
        require(counts["max_pto_backoff"] == max((t["backoff"] for t in timeouts), default=0),
                "PTO backoff maximum differs")
        require(counts["max_pto_in_flight_bytes"] ==
                max((t["in_flight_bytes"] for t in timeouts), default=0), "PTO flight maximum differs")
        if samples:
            previous = samples[-1]
            require(sample["elapsed_ns"] >= previous["elapsed_ns"] and
                    sample["observed_unix_ns"] >= previous["observed_unix_ns"], "trace clocks decrease")
            require(all(counts[k] >= previous["counts"][k] for k in COUNTS), "recovery counters decrease")
            for key in ("phases", "timeouts"):
                require(sample[key][:len(previous[key])] == previous[key], "event history changes")
        samples.append(sample)
    require(bool(samples) and samples[0]["observation"] == "start", "trace start is absent")
    terminal = [s for s in samples if s["observation"] in ("success", "failure")]
    require(len(terminal) <= 1, "multiple primary outcomes")
    if success:
        require(len(terminal) == 1 and terminal[0]["observation"] == "success", "success lacks trace outcome")
        require(terminal[0]["phase"] == "complete" and terminal[0]["counts"]["established"] == 1,
                "success lacks authenticated completion")
        require([p["phase"] for p in terminal[0]["phases"]] == list(PHASES[:-1]),
                "success lacks complete phase inventory")
        require(samples[-1]["observation"] == "drained", "success lacks endpoint drain")
    else:
        require(not terminal or terminal[0]["observation"] == "failure" or
                samples[-1]["observation"] == "drain_failure", "failure contains trace success")
    return {"observations": len(samples), "primary_outcome_observed": bool(terminal),
            "before_drain": terminal[0] if terminal else None, "last": samples[-1]}


def trace_audit(shards):
    result = audit(shards)
    observed = any("engine_observation" in s for s in shards)
    if observed:
        import hashlib
        from quic_observation.audit import state_trace, state_pair
        from quic_observation.prepare import verify_provenance
        root = Path(__file__).resolve().parents[1]
        patch = hashlib.sha256((root/"scripts/quic_observation/engine.patch").read_bytes()).hexdigest()
        require(all(s.get("engine_observation") == shards[0].get("engine_observation")
                    for s in shards), "mixed diagnostic engine provenance")
        provenance = shards[0]["engine_observation"]
        verify_provenance(provenance, root)
        require(provenance["patch_sha256"] == patch and provenance["version"] == "0.11.19",
                "diagnostic patch identity differs")
        require(all(s["lock_sha256"] == provenance["canonical_lock_sha256"] for s in shards),
                "diagnostic canonical lock differs")
        result["engine_observation"] = provenance
        result["engine_states"] = []
    require(all(s.get("recovery_trace") is True for s in shards), "tracing mode is not recorded")
    for shard in shards:
        preflight = shard.get("trace_preflight", {})
        require("reports" in preflight and "error" not in preflight, "trace preflight is absent/failed")
        from quic_profiles import verify
        for role, responder in (("client", False), ("server", True)):
            verify(preflight["reports"][role], responder, 1, 24)
            require(preflight["reports"][role]["congestion_policy"] == "cubic", "preflight policy differs")
            protocol_trace(diagnostics(preflight, role), True)
            if observed:
                state_trace(diagnostics(preflight, role))
    traces = []
    for shard in shards:
        for record in shard["runs"]:
            for flow in record.get("flows", [record]):
                traces.append({"suite": shard["suite"], "profile": record["profile"]["name"],
                               "repetition": record["repetition"], "streams": record["streams"],
                               "bytes": record["bytes"], "policy": flow["policy"],
                               "status": "failed" if "error" in flow else "passed",
                               **{role: protocol_trace(diagnostics(flow, role), "error" not in flow)
                                  for role in ("client", "server")}})
                if observed:
                    result["engine_states"].append({
                        **{k: traces[-1][k] for k in ("suite", "profile", "repetition", "streams",
                                                     "bytes", "policy", "status")},
                        **state_pair(diagnostics(flow, "client"), diagnostics(flow, "server"),
                                     "error" not in flow)})
    result["protocol_traces"] = traces
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())
              for p in args.shards]
    result = trace_audit(shards)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
