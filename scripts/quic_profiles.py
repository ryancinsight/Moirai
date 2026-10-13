#!/usr/bin/env python3
"""Measure the ADR 0075 workload using real mutually authenticated processes.

Only task-owned Linux namespaces receive netem rules. The same process/byte/
metrics verifier runs on loopback for native platform evidence. Measurements
are controlled profiles; no TCP comparison or wide-area equivalence is implied.
"""

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time
import uuid


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    delay_ms: int = 0
    loss: str = "0%"
    rate: str | None = None


PROFILES = [Profile("unshaped"), Profile("delay-10ms", 10, rate="100mbit"),
            Profile("delay-50ms", 50, rate="100mbit"),
            Profile("delay-10ms-loss-1pct", 10, "1%", "100mbit")]
STREAMS = [1, 4, 16]
FRAME_BYTES = [24, 4096, 65560]
MESSAGES = 128
REPETITIONS = 3
SEED = 20261013


def execute(arguments, **options):
    return subprocess.run(arguments, check=True, text=True, capture_output=True,
                          timeout=options.pop("timeout", 60), **options).stdout


class Network:
    def __init__(self, shaped):
        self.shaped = shaped
        self.names = [f"moirai-quic-{uuid.uuid4().hex[:12]}-{role}" for role in ("client", "server")]
        self.created = []
        self.address = "192.0.2.2:4433" if shaped else "127.0.0.1:0"

    def prefix(self, role):
        return ["sudo", "ip", "netns", "exec", self.names[role]] if self.shaped else []

    def setup(self):
        if not self.shaped:
            return
        if platform.system() != "Linux":
            raise RuntimeError("isolated netem profiles require Linux namespace administration")
        for name in self.names:
            execute(["sudo", "ip", "netns", "add", name])
            self.created.append(name)
        execute(["sudo", "ip", "link", "add", "quic-client", "netns", self.names[0],
                 "type", "veth", "peer", "name", "quic-server", "netns", self.names[1]])
        for role, interface, address in [(0, "quic-client", "192.0.2.1/24"),
                                         (1, "quic-server", "192.0.2.2/24")]:
            prefix = self.prefix(role)
            execute([*prefix, "ip", "link", "set", "lo", "up"])
            execute([*prefix, "ip", "address", "add", address, "dev", interface])
            execute([*prefix, "ip", "link", "set", interface, "up"])

    def shape(self, profile):
        if not self.shaped:
            return
        for role, interface in [(0, "quic-client"), (1, "quic-server")]:
            options = ["netem", "seed", str(SEED)]
            if profile.delay_ms:
                options.extend(["delay", f"{profile.delay_ms}ms"])
            if profile.loss != "0%":
                options.extend(["loss", profile.loss])
            if profile.rate:
                options.extend(["rate", profile.rate])
            execute([*self.prefix(role), "tc", "qdisc", "replace", "dev", interface,
                     "root", *options])

    def counters(self):
        if not self.shaped:
            return None
        return [json.loads(execute([*self.prefix(role), "tc", "-s", "-j", "qdisc", "show", "dev", interface]))
                for role, interface in [(0, "quic-client"), (1, "quic-server")]]

    def close(self):
        for name in reversed(self.created):
            execute(["sudo", "ip", "netns", "delete", name])


def verify(report, responder, streams, size):
    assert report["schema"] == 1 and report["responder"] is responder
    assert report["streams"] == streams and report["messages_per_stream"] == MESSAGES
    assert report["frame_bytes"] == size
    assert report["verified_payload_bytes_per_direction"] == streams * MESSAGES * size
    samples = report["samples_ns"]
    assert len(samples) == streams * MESSAGES and samples == sorted(samples)
    assert all(type(sample) is int and sample >= 0 for sample in samples)
    assert report["exchange_ns"] > 0 and report["connection_setup_ns"] > 0
    assert report["goodput_bits_per_second"] == streams * MESSAGES * size * 8 * 10**9 // report["exchange_ns"]
    for percent in [50, 95]:
        assert report[f"p{percent}_ns"] == samples[(percent * len(samples) + 99) // 100 - 1]


def pair(binary, network, streams, size):
    deadline = time.monotonic() + 60
    command = [str(binary), "server", network.address, str(streams), str(MESSAGES), str(size)]
    server = subprocess.Popen([*network.prefix(1), *command], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ready = queue.Queue()
    reader = threading.Thread(target=lambda: ready.put(server.stdout.readline()), daemon=True)
    reader.start()
    try:
        line = ready.get(timeout=min(5, deadline - time.monotonic()))
        address = json.loads(line)["ready"]
        client = execute([*network.prefix(0), str(binary), "client", address,
                          str(streams), str(MESSAGES), str(size)], timeout=deadline - time.monotonic())
        stdout, stderr = server.communicate(timeout=deadline - time.monotonic())
        if server.returncode != 0:
            raise RuntimeError(f"QUIC server fails: {stderr}")
        reports = {"client": json.loads(client), "server": json.loads(stdout)}
        verify(reports["client"], False, streams, size)
        verify(reports["server"], True, streams, size)
        return reports
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate(timeout=5)
        reader.join(timeout=5)
        if reader.is_alive():
            raise RuntimeError("task-owned readiness reader did not terminate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", choices=["loopback", "netem"], default="loopback")
    arguments = parser.parse_args()
    network = Network(arguments.network == "netem")
    report = {"schema": 1, "network": arguments.network, "runner": platform.platform(),
              "revision": execute(["git", "rev-parse", "HEAD"]).strip(),
              "dirty": bool(execute(["git", "status", "--porcelain"])),
              "rustc": execute(["rustc", "-Vv"]), "seed": SEED, "runs": []}
    if network.shaped:
        report["iproute2"] = execute(["tc", "-V"])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(network.close)
        network.setup()
        for profile in PROFILES if network.shaped else PROFILES[:1]:
            for streams in STREAMS:
                for size in FRAME_BYTES:
                    for repetition in range(REPETITIONS):
                        network.shape(profile)
                        record = {"profile": dataclasses.asdict(profile), "repetition": repetition,
                                  "streams": streams, "bytes": size, "before": network.counters()}
                        try:
                            record["reports"] = pair(arguments.binary.resolve(strict=True), network, streams, size)
                        except Exception as error:
                            record["error"] = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            record["after"] = network.counters()
                            report["runs"].append(record)
                            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(profile.name, streams, size, repetition, "verified", flush=True)


if __name__ == "__main__":
    main()
