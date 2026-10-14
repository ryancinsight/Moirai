#!/usr/bin/env python3
"""Matched stock-controller experiments; no result changes the promotion gate.

Each matrix shard is one of the three original repetitions. Together they
retain every one of the 108 original pairs per policy. Fairness supplements
that matrix with two maximum-load pairs sharing the same kernel queues.
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import threading
import time

from quic_profiles import (FRAME_BYTES, MESSAGES, PROFILES, REPETITIONS, SEED,
                           STREAMS, Network, execute, pair)

POLICIES = ("cubic", "newreno", "bbr-experimental")
COMPETITORS = (("cubic", "cubic"), ("newreno", "cubic"),
               ("newreno", "newreno"), ("bbr-experimental", "cubic"),
               ("bbr-experimental", "bbr-experimental"))


class QueueSamples:
    """One observer per experiment, identical across policies, including errors."""

    def __init__(self, network):
        self.network = network
        self.stop = threading.Event()
        self.samples = []
        self.error = None
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def observe(self):
        try:
            while not self.stop.is_set():
                self.samples.append({"observed_unix_ns": time.time_ns(),
                                     "counters": self.network.counters()})
                self.stop.wait(1)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("queue observer exceeds its bounded shutdown")
        if self.error:
            raise RuntimeError(self.error)


def failure_record(error):
    result = {"error": f"{type(error).__name__}: {error}"}
    for field in ("server_stdout", "server_stderr", "stdout", "stderr"):
        value = getattr(error, field, None)
        result[field] = value.decode(errors="replace") if isinstance(value, bytes) else value
    return result


def record_pair(binary, network, streams, size, policy, **options):
    try:
        return {"policy": policy, "reports": pair(binary, network, streams, size,
                                                  policy=policy, **options)}
    except Exception as error:
        return {"policy": policy, **failure_record(error)}


def matrix_cases(repetition):
    policies = POLICIES[repetition:] + POLICIES[:repetition]
    for profile in PROFILES:
        for streams in STREAMS:
            for size in FRAME_BYTES:
                for policy in policies:
                    yield profile, streams, size, policy


def matrix(binary, network, repetition):
    for profile, streams, size, policy in matrix_cases(repetition):
        network.shape(profile)
        record = {"profile": dataclasses.asdict(profile), "streams": streams,
                  "bytes": size, "repetition": repetition,
                  "before": network.counters()}
        with QueueSamples(network) as queues:
            record.update(record_pair(binary, network, streams, size, policy,
                                      cleanup_namespace=False))
        # Join the observer before namespace-wide process cleanup. Its short
        # kernel counter readers are not transport processes to terminate.
        network.stop(0)
        network.stop(1)
        record["queue_samples"] = queues.samples
        record["after"] = network.counters()
        yield record


def fairness(binary, network, repetition):
    competitors = COMPETITORS[repetition:] + COMPETITORS[:repetition]
    for profile in (PROFILES[1], PROFILES[3]):
        for policies in competitors:
            network.shape(profile)
            record = {"profile": dataclasses.asdict(profile), "streams": 16,
                      "bytes": 65_560, "repetition": repetition,
                      "before": network.counters()}
            barrier = threading.Barrier(2)
            with QueueSamples(network) as queues:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
                    tasks = [workers.submit(record_pair, binary, network, 16, 65_560,
                                             policy, address=f"192.0.2.2:{4433 + index}",
                                             barrier=barrier, cleanup_namespace=False)
                             for index, policy in enumerate(policies)]
                    record["flows"] = [task.result(timeout=70) for task in tasks]
            network.stop(0)
            network.stop(1)
            record["queue_samples"] = queues.samples
            record["after"] = network.counters()
            yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["matrix", "fairness"], required=True)
    parser.add_argument("--repetition", type=int, choices=range(REPETITIONS), required=True)
    arguments = parser.parse_args()
    network = Network(True, arguments.tc)
    report = {"schema": 1, "status": "running", "suite": arguments.suite,
              "runner": platform.platform(), "revision": execute(["git", "rev-parse", "HEAD"]).strip(),
              "tree": execute(["git", "rev-parse", "HEAD^{tree}"]).strip(),
              "dirty": bool(execute(["git", "status", "--porcelain"])),
              "rustc": execute(["rustc", "-Vv"]), "seed": SEED,
              "iproute2": execute([network.tc, "-V"]), "repetition": arguments.repetition,
              "binary_sha256": hashlib.sha256(arguments.binary.read_bytes()).hexdigest(),
              "lock_sha256": hashlib.sha256(Path("Cargo.lock").read_bytes()).hexdigest(),
              "policies": POLICIES, "messages_per_stream": MESSAGES,
              "pair_deadline_seconds": 60, "operation_deadline_seconds": 5,
              "runs": []}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(network.close)
            network.setup()
            experiment = matrix if arguments.suite == "matrix" else fairness
            for record in experiment(arguments.binary.resolve(strict=True), network, arguments.repetition):
                report["runs"].append(record)
                save()
                failed = "error" in record or any("error" in flow for flow in record.get("flows", []))
                print(arguments.suite, record["profile"]["name"], record.get("policy", "competing"),
                      record["streams"], record["bytes"], "failed" if failed else "verified", flush=True)
            expected = 108 if arguments.suite == "matrix" else 10
            if len(report["runs"]) != expected:
                raise RuntimeError("experiment inventory is incomplete")
            failures = sum("error" in flow for record in report["runs"]
                           for flow in record.get("flows", [record]))
            report["failed_pairs"] = failures
            if failures:
                raise RuntimeError(f"{failures} process pairs fail; all experiment records remain failures")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    else:
        report["status"] = "passed"
    finally:
        save()


if __name__ == "__main__":
    main()
