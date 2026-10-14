#!/usr/bin/env python3
"""Independently verify recovery shards and summarize complete/failed evidence.

All arithmetic uses integers or exact fractions. Sampled queue maxima and
sampled-progress fairness are observations, not continuous-time bounds.
"""

import argparse
import dataclasses
from fractions import Fraction
import gzip
import itertools
import json
from pathlib import Path
import re

from quic_profiles import FRAME_BYTES, MESSAGES, PROFILES, REPETITIONS, SEED, STREAMS, verify
from quic_recovery import COMPETITORS, POLICIES

UDP_ERRORS = ("InErrors", "RcvbufErrors", "SndbufErrors", "InCsumErrors", "MemErrors")
FIELDS = ("observed_unix_ns", "exchange_elapsed_ns", "completed_messages", "lost_packets",
          "congestion_events", "congestion_window_bytes", "round_trip_ns", "path_mtu")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def progress(trace):
    samples = []
    for line in (trace or "").splitlines():
        if "verified progress and recovery snapshot" not in line:
            continue
        sample = {}
        for field in FIELDS:
            match = re.search(rf"\b{field}=(\d+)", line)
            require(match is not None, f"missing progress field {field}")
            sample[field] = int(match[1])
        latency = re.search(r"completed_cycle_p95_ns=Some\((\d+)\)", line)
        sample["completed_cycle_p95_ns"] = int(latency[1]) if latency else None
        if samples:
            for field in ("observed_unix_ns", "exchange_elapsed_ns", "completed_messages",
                          "lost_packets", "congestion_events"):
                require(sample[field] >= samples[-1][field], f"nonmonotonic {field}")
        samples.append(sample)
    return samples


def diagnostics(flow, role):
    if "reports" in flow:
        return flow["reports"]["diagnostics"][f"{role}_stderr"]
    return flow.get("server_stderr" if role == "server" else "stderr")


def fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def queue_evidence(record):
    samples = record["queue_samples"]
    require(bool(samples), "queue sampling is absent")
    profile = record["profile"]
    for snapshot in [record["before"], record["after"], *[s["counters"] for s in samples]]:
        require(len(snapshot) == 2, "both queue directions are required")
        for peer in snapshot:
            require(len(peer["qdisc"]) == 1 and peer["qdisc"][0]["kind"] == "netem", "different qdisc")
            options = peer["qdisc"][0]["options"]
            require(options["seed"] == SEED and options["limit"] == 1000, "changed queue seed/limit")
            require(Fraction(str(options.get("delay", {}).get("delay", 0))) ==
                    Fraction(profile["delay_ms"], 1000), "actual queue delay mismatch")
            require(Fraction(str(options.get("loss-random", {}).get("loss", 0))) ==
                    Fraction(profile["loss"].rstrip("%")) / 100, "actual loss mismatch")
            require(options.get("rate", {}).get("rate") ==
                    (100_000_000 // 8 if profile["rate"] == "100mbit" else None), "actual rate mismatch")
    result = []
    for role in range(2):
        points = [(sample["observed_unix_ns"], sample["counters"][role]["qdisc"][0]["backlog"])
                  for sample in samples]
        before, after = record["before"][role], record["after"][role]
        deltas = {field: after["udp"][field] - before["udp"][field] for field in UDP_ERRORS}
        require(all(value >= 0 for value in deltas.values()), "UDP counters decrease")
        queue_bytes = after["qdisc"][0]["bytes"] - before["qdisc"][0]["bytes"]
        flows = record.get("flows", [record])
        completed = sum(record["streams"] * MESSAGES * record["bytes"]
                        for flow in flows if "error" not in flow)
        require(queue_bytes >= completed, "kernel queue bytes fail to cover complete payloads")
        xs = [time - points[0][0] for time, _ in points]
        ys = [size for _, size in points]
        require(all(a < b for a, b in zip(xs, xs[1:])), "queue sample clocks do not increase")
        n = len(points)
        denominator = n * sum(x * x for x in xs) - sum(xs) ** 2
        slope = (Fraction((n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) * 10**9,
                          denominator) if denominator else None)
        result.append({"samples": n, "observed_max_backlog_bytes": max(ys),
                       "first_backlog_bytes": ys[0], "last_backlog_bytes": ys[-1],
                       "after_backlog_bytes": after["qdisc"][0]["backlog"],
                       "sampled_backlog_slope_bytes_per_second": fraction(slope) if slope is not None else None,
                       "kernel_queue_bytes": queue_bytes, "udp_error_deltas": deltas})
    return result


def sampled_fairness(flows, size):
    """Jain index over observed validated-counter increments in a common span.

    Last-observation counters have up to one sampling interval of staleness;
    retain actual boundary ages instead of interpolating unobserved progress.
    """
    traces = [progress(diagnostics(flow, "client")) for flow in flows]
    if any(len(trace) < 2 for trace in traces):
        return {"unavailable": "fewer than two progress observations per flow"}
    start = max(trace[0]["observed_unix_ns"] for trace in traces)
    end = min(trace[-1]["observed_unix_ns"] for trace in traces)
    if end <= start:
        return {"unavailable": "no shared observed interval"}
    increments, ages = [], []
    for trace in traces:
        first = max((p for p in trace if p["observed_unix_ns"] <= start),
                    key=lambda p: p["observed_unix_ns"])
        last = max((p for p in trace if p["observed_unix_ns"] <= end),
                   key=lambda p: p["observed_unix_ns"])
        increments.append((last["completed_messages"] - first["completed_messages"]) * size)
        ages.append({"start_ns": start - first["observed_unix_ns"],
                     "end_ns": end - last["observed_unix_ns"]})
    square_sum = sum(value * value for value in increments)
    return {"observed_start_unix_ns": start, "observed_end_unix_ns": end,
            "verified_counter_byte_increments": increments, "boundary_sample_ages": ages,
            "jain_sampled_progress": fraction(Fraction(sum(increments)**2, len(flows) * square_sum))
            if square_sum else None}


def check_flow(flow, record):
    require(flow["policy"] in POLICIES, "unknown policy")
    if "error" in flow:
        require("reports" not in flow, "failure contains a successful partial report")
    else:
        for role, responder in (("client", False), ("server", True)):
            report = flow["reports"][role]
            verify(report, responder, record["streams"], record["bytes"])
            require(report["congestion_policy"] == flow["policy"], "measured policy mismatch")
    result = {"policy": flow["policy"], "status": "failed" if "error" in flow else "passed"}
    if "error" not in flow:
        report = flow["reports"]["client"]
        result.update({key: report[key] for key in
                       ("exchange_ns", "goodput_bits_per_second", "p50_ns", "p95_ns")})
    else:
        result["error"] = flow["error"]
    for role in ("client", "server"):
        samples = progress(diagnostics(flow, role))
        require(all(p["completed_messages"] <= record["streams"] * MESSAGES for p in samples),
                "progress exceeds the original workload")
        result[f"{role}_last_progress"] = samples[-1] if samples else None
    return result


def audit(shards):
    require(len(shards) == 2 * REPETITIONS, "all six matrix/fairness shards are required")
    require(len({s["revision"] for s in shards}) == 1, "mixed source revisions")
    require(len({s["tree"] for s in shards}) == 1, "mixed source trees")
    require(len({s["lock_sha256"] for s in shards}) == 1, "mixed dependency locks")
    require(len({s["runner"] for s in shards}) == 1, "different kernel/runner platforms")
    require(len({s["iproute2"] for s in shards}) == 1, "different netem tools")
    require({(s["suite"], s["repetition"]) for s in shards} ==
            set(itertools.product(("matrix", "fairness"), range(REPETITIONS))), "shard inventory mismatch")
    matrix, competing = [], []
    expected_profiles = {p.name: dataclasses.asdict(p) for p in PROFILES}
    for shard in shards:
        require(shard["schema"] == 1 and shard["status"] in ("passed", "failed"), "unfinished/unknown shard")
        require(shard["seed"] == SEED and not shard["dirty"], "seed mismatch or dirty measured source")
        require(shard["policies"] == list(POLICIES), "changed controller inventory")
        require(shard["messages_per_stream"] == MESSAGES, "changed message count")
        require(shard["pair_deadline_seconds"] == 60 and shard["operation_deadline_seconds"] == 5,
                "changed deadlines")
        failed_pairs = sum("error" in flow for record in shard["runs"]
                           for flow in record.get("flows", [record]))
        require(shard["failed_pairs"] == failed_pairs, "failure count mismatch")
        require(shard["status"] == ("failed" if failed_pairs else "passed"), "failure status mismatch")
        for record in shard["runs"]:
            profile = record["profile"]["name"]
            require(record["profile"] == expected_profiles[profile], "changed profile")
            require(record["repetition"] == shard["repetition"], "changed repetition")
            base = {key: record[key] for key in ("profile", "streams", "bytes", "repetition")}
            base["queues"] = queue_evidence(record)
            if shard["suite"] == "matrix":
                base.update(check_flow(record, record))
                matrix.append(base)
            else:
                require(record["streams"] == 16 and record["bytes"] == 65_560, "changed competing workload")
                base["flows"] = [check_flow(flow, record) for flow in record["flows"]]
                base["sampled_fairness"] = sampled_fairness(record["flows"], record["bytes"])
                competing.append(base)
    expected = set(itertools.product(expected_profiles, STREAMS, FRAME_BYTES, range(REPETITIONS)))
    counts = {}
    for policy in POLICIES:
        cases = [r for r in matrix if r["policy"] == policy]
        require(len(cases) == len(expected), f"incomplete matrix for {policy}")
        require({(r["profile"]["name"], r["streams"], r["bytes"], r["repetition"])
                 for r in cases} == expected, "duplicate or missing matrix cases")
        successes = [r for r in cases if r["status"] == "passed"]
        counts[policy] = {"pairs": len(cases), "passed": len(successes), "failed": len(cases) - len(successes),
                          "complete_payload_bytes_per_direction":
                          sum(r["streams"] * MESSAGES * r["bytes"] for r in successes)}
    expected_competitors = set(itertools.product((PROFILES[1].name, PROFILES[3].name),
                                                COMPETITORS, range(REPETITIONS)))
    require(len(competing) == len(expected_competitors), "incomplete competing-flow inventory")
    require({(r["profile"]["name"], tuple(f["policy"] for f in r["flows"]), r["repetition"])
             for r in competing} == expected_competitors, "missing/duplicate competing-flow cases")
    return {"revision": shards[0]["revision"], "tree": shards[0]["tree"],
            "runner": shards[0]["runner"], "iproute2": shards[0]["iproute2"],
            "counts": counts, "matrix": matrix, "competing": competing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())
              for p in args.shards]
    result = audit(shards)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
