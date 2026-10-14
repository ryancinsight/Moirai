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
from pathlib import Path
import platform
import queue
import subprocess
import sys
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
    result = subprocess.run(arguments, text=True, capture_output=True,
                            timeout=options.pop("timeout", 60), **options)
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    return result.stdout


class Network:
    def __init__(self, shaped, tc="tc"):
        self.shaped = shaped
        self.tc = tc
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
            execute([*self.prefix(role), self.tc, "qdisc", "replace", "dev", interface,
                     "root", *options])

    def counters(self):
        if not self.shaped:
            return None
        counters = []
        for role, interface in [(0, "quic-client"), (1, "quic-server")]:
            prefix = self.prefix(role)
            lines = execute([*prefix, "cat", "/proc/net/snmp"]).splitlines()
            udp = [line.split()[1:] for line in lines if line.startswith("Udp:")]
            if len(udp) != 2 or len(udp[0]) != len(udp[1]):
                raise RuntimeError("namespace UDP counter schema is invalid")
            counters.append({"qdisc": json.loads(execute([*prefix, self.tc, "-s", "-j", "qdisc", "show", "dev", interface])),
                             "udp": dict(zip(udp[0], map(int, udp[1]), strict=True)),
                             "sockets": execute([*prefix, "ss", "-u", "-a", "-n", "-m"])})
        return counters

    def stop(self, role):
        if self.shaped and self.names[role] in self.created:
            processes = execute(["sudo", "ip", "netns", "pids", self.names[role]]).split()
            if any(not value.isdecimal() or int(value) <= 1 for value in processes):
                raise RuntimeError("owned namespace process inventory is invalid")
            if processes:
                execute(["sudo", "kill", "-KILL", *processes])

    def close(self):
        for role in range(len(self.names)):
            self.stop(role)
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


def pair(binary, network, streams, size, *, policy="cubic", address=None, barrier=None):
    deadline = time.monotonic() + 60
    command = [str(binary), "server", address or network.address, str(streams), str(MESSAGES), str(size), policy]
    server = subprocess.Popen([*network.prefix(1), *command], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ready = queue.Queue()
    reader = threading.Thread(target=lambda: ready.put(server.stdout.readline()), daemon=True)
    reader.start()
    try:
        line = ready.get(timeout=min(5, deadline - time.monotonic()))
        address = json.loads(line)["ready"]
        if barrier is not None:
            barrier.wait(timeout=min(5, deadline - time.monotonic()))
        client = subprocess.run([*network.prefix(0), str(binary), "client", address,
                                 str(streams), str(MESSAGES), str(size), policy], text=True,
                                capture_output=True, timeout=deadline - time.monotonic())
        client.check_returncode()
        stdout, stderr = server.communicate(timeout=deadline - time.monotonic())
        if server.returncode != 0:
            raise RuntimeError(f"QUIC server fails: {stderr}")
        reports = {"client": json.loads(client.stdout), "server": json.loads(stdout)}
        verify(reports["client"], False, streams, size)
        verify(reports["server"], True, streams, size)
        assert all(report["congestion_policy"] == policy for report in reports.values())
        reports["diagnostics"] = {"client_stderr": client.stderr, "server_stderr": stderr}
        return reports
    finally:
        failure = sys.exception()
        # A shared-queue experiment owns several pairs in these namespaces.
        # The controller drains/kills its namespace only after all pairs finish.
        if barrier is None:
            network.stop(0)
            network.stop(1)
        if server.poll() is None:
            server.terminate()
            try:
                stdout, stderr = server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                stdout, stderr = server.communicate(timeout=5)
        else:
            stdout, stderr = server.communicate(timeout=5)
        if failure is not None:
            failure.server_stdout = stdout
            failure.server_stderr = stderr
        reader.join(timeout=5)
        if reader.is_alive():
            raise RuntimeError("task-owned readiness reader did not terminate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", choices=["loopback", "netem"], default="loopback")
    parser.add_argument("--tc", default="tc", help="Exact seeded-netem tool selected by the caller")
    parser.add_argument("--policy", choices=["cubic", "newreno", "bbr-experimental"], default="cubic")
    parser.add_argument("--case", default="", help="Diagnostic PROFILE:STREAMS:FRAME_BYTES; default verifies the full matrix")
    arguments = parser.parse_args()
    cases = [(profile, streams, size) for profile in PROFILES if arguments.network == "netem" or profile == PROFILES[0]
             for streams in STREAMS for size in FRAME_BYTES]
    if arguments.case:
        cases = [(profile, streams, size) for profile, streams, size in cases
                 if arguments.case == f"{profile.name}:{streams}:{size}"]
        if len(cases) != 1:
            parser.error("case must identify one member of the declared profile/stream/frame domain")
    network = Network(arguments.network == "netem", arguments.tc)
    report = {"schema": 1, "status": "running", "network": arguments.network, "runner": platform.platform(),
              "revision": execute(["git", "rev-parse", "HEAD"]).strip(),
              "dirty": bool(execute(["git", "status", "--porcelain"])),
              "rustc": execute(["rustc", "-Vv"]), "seed": SEED, "selected_case": arguments.case,
              "expected_pairs": len(cases) * REPETITIONS, "congestion_policy": arguments.policy, "runs": []}
    if network.shaped:
        report["iproute2"] = execute([network.tc, "-V"])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(network.close)
            network.setup()
            for profile, streams, size in cases:
                for repetition in range(REPETITIONS):
                    network.shape(profile)
                    record = {"profile": dataclasses.asdict(profile), "repetition": repetition,
                              "streams": streams, "bytes": size, "before": network.counters()}
                    try:
                        record["reports"] = pair(arguments.binary.resolve(strict=True), network, streams, size, policy=arguments.policy)
                    except Exception as error:
                        record["error"] = f"{type(error).__name__}: {error}"
                        record["server_stdout"] = getattr(error, "server_stdout", None)
                        record["server_stderr"] = getattr(error, "server_stderr", None)
                        if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                            for field in ["stdout", "stderr"]:
                                value = getattr(error, field)
                                record[f"error_{field}"] = value.decode(errors="replace") if isinstance(value, bytes) else value
                    finally:
                        record["after"] = network.counters()
                        report["runs"].append(record)
                        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(profile.name, streams, size, repetition, "failed" if "error" in record else "verified", flush=True)
            failures = sum("error" in record for record in report["runs"])
            if failures:
                raise RuntimeError(f"{failures} of {report['expected_pairs']} process pairs fail; complete failure evidence is preserved")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            report["error_stdout"] = error.stdout
            report["error_stderr"] = error.stderr
        raise
    else:
        report["status"] = "passed"
    finally:
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
