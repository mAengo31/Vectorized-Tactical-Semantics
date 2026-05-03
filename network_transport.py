"""
network_transport.py — single-channel STRE transport over UDP/WiFi.

Transport choice rationale:

The CASK system operates on a tactical network model: lossy radio link,
limited bandwidth, no head-of-line blocking acceptable. UDP fits this
perfectly. TCP would add retransmit logic that violates real-time
guarantees — a 5-second-old Type 1 observation is worse than no observation
because it implies the operator is looking at stale truth.

Therefore: one UDP datagram per sealed STRE message. Edge sends best-effort.
C2 receives, unseals, decodes, ingests into C2StreEngine.

What crosses the network:
    Only sealed STRE Type 1/Type 2 envelopes — the same bytes that get
    written to wire_log.bin locally. Average 67-72 bytes sealed plus
    28 bytes of UDP/IP framing = ~95-100 bytes per packet.

What does NOT cross the network:
    Pixel bboxes. Raw labels from grounding_dino. Scene captions.
    Annotated frames. foundry_events.jsonl. Anything that the spec does
    not put on the wire. Those are local edge artifacts for forensic
    audit, retrieved out-of-band post-mission.

Why one channel only:
    C2 makes decisions from semantic events, not from looking at edge's
    raw video. The pixel data has no role in the C2 decision loop. Sending
    it would (a) violate the bandwidth-discipline story, (b) introduce a
    second transport with its own failure modes, and (c) let C2 second-guess
    the edge's perception, which defeats the purpose of edge AI.

Kill-switch behavior:
    Edge: keeps sealing and writing wire_log.bin locally regardless of
          network state. UDP packets blackhole during outage. Counter
          continues advancing.
    C2:   silence during outage. On recovery, observes a counter gap
          (e.g. last seen ctr=247, next received ctr=312 → 64 messages lost).
          Logs the gap. Resumes ingestion from the new counter. Does NOT
          replay or request retransmission — those Type 1s described
          real-time positions and are now stale.

Anti-replay & forgery:
    Counter is monotonic, persisted across reboots in ~/.cask/cask.counter.
    Any received message with counter ≤ last-seen is rejected.
    Sealed payloads use ChaCha20-Poly1305 with shared 256-bit PSK from
    ~/.cask/cask.psk. Without the PSK, an adversary cannot forge messages
    or decrypt past traffic.

PSK distribution (for the demo):
    Generate once on edge, copy to C2 over a secure side-channel (USB,
    scp during setup). Both processes look in ~/.cask/cask.psk by default.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# ============================================================================
# Persistent shared key + counter
# ============================================================================

DEFAULT_CONFIG_DIR = Path.home() / ".cask"
PSK_FILENAME = "cask.psk"
COUNTER_FILENAME = "cask.counter"


def get_or_create_psk(config_dir: Optional[Path] = None,
                      regenerate: bool = False) -> bytes:
    """
    Load the persistent ChaCha20-Poly1305 PSK from disk, or create one
    if it doesn't exist.

    The file at ~/.cask/cask.psk is the system's identity. Both edge and
    C2 must have the same bytes there. Distribute out-of-band when first
    setting up a new C2/edge pair.

    Args:
        config_dir: where to look for cask.psk. Defaults to ~/.cask/
        regenerate: if True, overwrite any existing PSK with a new random
                    one. Useful for "rotate the key" workflows. Don't pass
                    True unless you also plan to redistribute.

    Returns:
        32 raw bytes (256-bit key for ChaCha20-Poly1305).
    """
    cfg = config_dir or DEFAULT_CONFIG_DIR
    cfg.mkdir(parents=True, exist_ok=True)
    psk_path = cfg / PSK_FILENAME

    if psk_path.exists() and not regenerate:
        psk = psk_path.read_bytes()
        if len(psk) != 32:
            raise ValueError(
                f"PSK at {psk_path} is {len(psk)} bytes, expected 32. "
                f"Delete it and regenerate, then redistribute to all "
                f"CASK and C2 nodes."
            )
        return psk

    import secrets
    psk = secrets.token_bytes(32)
    psk_path.write_bytes(psk)
    try:
        os.chmod(psk_path, 0o600)
    except OSError:
        pass
    print(f"[psk] Generated new PSK at {psk_path}")
    print(f"[psk]   Distribute this file to C2 and any other CASKs out-of-band.")
    print(f"[psk]   scp {psk_path} user@c2-host:~/.cask/cask.psk")
    return psk


def get_persistent_counter_path(config_dir: Optional[Path] = None) -> Path:
    """
    Return the path to the persistent anti-replay counter.

    The counter MUST persist across reboots. If it ever resets, an
    adversary who recorded wire_log.bin from a previous run can replay
    those messages — C2 has no way to reject them.
    """
    cfg = config_dir or DEFAULT_CONFIG_DIR
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg / COUNTER_FILENAME


# ============================================================================
# Edge-side: UDP transmitter
# ============================================================================

class StreUdpTransmitter:
    """
    Best-effort UDP send for sealed STRE envelopes.

    No retransmit, no acknowledgment. Network drops are detected at C2
    via counter gaps, not here. This matches tactical-radio semantics:
    the link is lossy by design; replaying old observations is wrong;
    the protocol tolerates loss.

    Use as a fan-out alongside the local file writer (see
    NetworkAwareStreSink below). The local file write must succeed
    regardless of network state.
    """

    def __init__(self, c2_host: str, c2_port: int = 9601,
                 send_timeout_s: float = 0.05):
        self.c2_host = c2_host
        self.c2_port = c2_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(send_timeout_s)
        self._stats = {
            "packets_sent": 0,
            "send_errors": 0,
            "bytes_sent": 0,
            "last_send_ok": True,
        }
        self._lock = threading.Lock()

    def send_sealed(self, sealed_bytes: bytes) -> bool:
        """
        Fire-and-forget. Returns True if the OS accepted the datagram for
        sending (not whether it arrived — that's UDP's character).
        """
        try:
            self._sock.sendto(sealed_bytes, (self.c2_host, self.c2_port))
            with self._lock:
                self._stats["packets_sent"] += 1
                self._stats["bytes_sent"] += len(sealed_bytes)
                self._stats["last_send_ok"] = True
            return True
        except (socket.timeout, OSError):
            with self._lock:
                self._stats["send_errors"] += 1
                self._stats["last_send_ok"] = False
            return False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        self._sock.close()


# ============================================================================
# Edge-side: drop-in StreSink wrapper that fans out to UDP
# ============================================================================

class NetworkAwareStreSink:
    """
    Wraps the existing StreSink. Every sealed message is:
      1. Written to wire_log.bin (UNCONDITIONAL — the local artifact must
         always reflect what edge perceived, regardless of net state)
      2. Sent over UDP to C2 (best-effort)

    Use as a drop-in replacement for StreSink in main_loop.py:

        from stre_codec import StreSink, CoseSealer
        from network_transport import StreUdpTransmitter, NetworkAwareStreSink

        local = StreSink(sealer, wire_log_path=str(out_dir / "wire_log.bin"))
        udp_tx = StreUdpTransmitter("10.1.62.79", 9601)  # C2's IP
        stre_sink = NetworkAwareStreSink(local, udp_tx)
    """

    def __init__(self, local_sink, udp_tx: Optional[StreUdpTransmitter] = None):
        self.local_sink = local_sink
        self.udp_tx = udp_tx

    def healthcheck(self) -> bool:
        return self.local_sink.healthcheck()

    def write_type1(self, obs) -> bytes:
        sealed = self.local_sink.write_type1(obs)
        if self.udp_tx is not None:
            self.udp_tx.send_sealed(sealed)
        return sealed

    def write_type2(self, inf) -> bytes:
        sealed = self.local_sink.write_type2(inf)
        if self.udp_tx is not None:
            self.udp_tx.send_sealed(sealed)
        return sealed

    def stats(self) -> Dict[str, Any]:
        s = dict(self.local_sink.stats())
        if self.udp_tx is not None:
            s.update({f"udp_{k}": v for k, v in self.udp_tx.stats().items()})
        return s


# ============================================================================
# C2-side: UDP listener
# ============================================================================

class StreUdpListener:
    """
    Listens on a UDP port for sealed STRE envelopes from one or more
    CASKs. Unseals each, parses the message type, and dispatches to
    user-supplied callbacks.

    Tracks counter state per source_id. Logs counter gaps (lost packets)
    and rejects replays (counter ≤ last seen). Per the spec, no
    retransmission requested.

    Run inside its own thread or as the main loop of a standalone
    c2_listener.py process.
    """

    def __init__(self, psk: bytes, bind_host: str = "0.0.0.0",
                 bind_port: int = 9601):
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        if len(psk) != 32:
            raise ValueError("PSK must be exactly 32 bytes")
        self._aead = ChaCha20Poly1305(psk)
        self.bind_host = bind_host
        self.bind_port = bind_port

        # Per-source counter tracking. Key is source_id (uint16).
        self._last_counter: Dict[int, int] = {}
        self._counter_gaps: Dict[int, int] = {}
        self._replays_rejected: Dict[int, int] = {}
        self._decrypt_failures = 0
        self._packets_received = 0
        self._bytes_received = 0
        self._last_seen_ts: Dict[int, float] = {}
        self._lock = threading.Lock()

        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()

    def serve(self,
              on_observation: Optional[Callable] = None,
              on_inference: Optional[Callable] = None) -> None:
        """
        Block on the UDP socket. Calls on_observation(obs, src_id, counter)
        for each Type 1, on_inference(inf, src_id, counter) for each Type 2.

        Run in its own thread or as main of a standalone process.
        """
        from stre_codec import (
            CoseEnvelope, StreObservation, StreInference,
        )

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow large receive buffer for bursty traffic
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self._sock.bind((self.bind_host, self.bind_port))
        self._sock.settimeout(0.5)

        print(f"[c2-listener] STRE UDP listener up on "
              f"{self.bind_host}:{self.bind_port}")

        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            with self._lock:
                self._packets_received += 1
                self._bytes_received += len(data)

            try:
                env = CoseEnvelope.from_bytes(data)
            except Exception:
                with self._lock:
                    self._decrypt_failures += 1
                continue

            # Anti-replay: counter must be strictly greater than last seen
            with self._lock:
                last = self._last_counter.get(env.source_id, 0)
                if env.counter <= last:
                    self._replays_rejected[env.source_id] = (
                        self._replays_rejected.get(env.source_id, 0) + 1
                    )
                    continue
                # Detect gap
                gap = env.counter - last - 1
                if last > 0 and gap > 0:
                    self._counter_gaps[env.source_id] = (
                        self._counter_gaps.get(env.source_id, 0) + gap
                    )
                    print(f"[c2-listener] counter gap from "
                          f"src=0x{env.source_id:04x}: "
                          f"missed {gap} messages "
                          f"(last={last}, now={env.counter})")
                self._last_counter[env.source_id] = env.counter
                self._last_seen_ts[env.source_id] = time.time()

            # Decrypt
            try:
                plaintext = self._aead.decrypt(env.nonce_bytes, env.ciphertext, None)
            except Exception:
                with self._lock:
                    self._decrypt_failures += 1
                continue

            # Decode message type — first byte of CBOR array's first
            # element is the message type code (0x01 for obs, 0x02 for inf)
            try:
                import cbor2
                arr = cbor2.loads(plaintext)
                if not isinstance(arr, list) or len(arr) < 1:
                    continue
                mtype = arr[0]
            except Exception:
                with self._lock:
                    self._decrypt_failures += 1
                continue

            if mtype == 0x01:
                obs = StreObservation.decode(plaintext)
                if on_observation:
                    try:
                        on_observation(obs, env.source_id, env.counter)
                    except Exception as e:
                        print(f"[c2-listener] on_observation handler "
                              f"raised: {e}")
            elif mtype == 0x02:
                inf = StreInference.decode(plaintext)
                if on_inference:
                    try:
                        on_inference(inf, env.source_id, env.counter)
                    except Exception as e:
                        print(f"[c2-listener] on_inference handler "
                              f"raised: {e}")

        self._sock.close()
        print("[c2-listener] stopped")

    def stop(self) -> None:
        self._stop.set()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "packets_received": self._packets_received,
                "bytes_received": self._bytes_received,
                "decrypt_failures": self._decrypt_failures,
                "by_source": {
                    f"0x{src:04x}": {
                        "last_counter": self._last_counter.get(src, 0),
                        "counter_gaps": self._counter_gaps.get(src, 0),
                        "replays_rejected": self._replays_rejected.get(src, 0),
                        "last_seen_ago_s": (
                            time.time() - self._last_seen_ts.get(src, 0)
                            if src in self._last_seen_ts else None
                        ),
                    }
                    for src in self._last_counter
                },
            }


# ============================================================================
# Smoke test
# ============================================================================

def _smoke_test():
    """End-to-end roundtrip on localhost. Useful before testing across
    real WiFi between Jetson and Mac."""
    import threading
    from stre_codec import (
        CoseSealer, AntiReplayState, StreObservation, StreInference,
        OBJECT_CLASS, SENSOR_TYPE, PATTERN_ID, ACTION, THREAT_LEVEL,
    )

    # Use a test PSK so we don't clobber a real one
    test_dir = Path("/tmp/cask_network_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    psk = get_or_create_psk(config_dir=test_dir, regenerate=True)

    # Counter state for edge sealer
    counter_state = AntiReplayState(
        source_id=0xCA51,
        state_path=str(test_dir / "test.counter"),
    )
    sealer = CoseSealer(psk, source_id=0xCA51, counter_state=counter_state)

    # Spin up listener in a thread
    received: list = []
    def on_obs(obs, src, ctr):
        received.append(("obs", obs, src, ctr))
    def on_inf(inf, src, ctr):
        received.append(("inf", inf, src, ctr))

    listener = StreUdpListener(psk, bind_host="127.0.0.1", bind_port=19601)
    t = threading.Thread(
        target=listener.serve,
        kwargs={"on_observation": on_obs, "on_inference": on_inf},
        daemon=True,
    )
    t.start()
    time.sleep(0.3)

    # Edge transmitter
    tx = StreUdpTransmitter("127.0.0.1", 19601)

    # Send 5 observations and 1 inference
    now_ts = int(time.time())
    for i in range(5):
        obs = StreObservation(
            event_id=i + 1,
            source_id=0xCA51,
            object_class=OBJECT_CLASS["vehicle_tracked"],
            sensor_type=SENSOR_TYPE["EO"],
            confidence=70 + i,
            lat=int(48.158500 * 1e6),
            lon=int(37.727000 * 1e6) + i * 100,
            alt_m=0,
            heading=180,
            speed_mps_x10=0,
            timestamp=now_ts,
        )
        sealed = sealer.seal(obs.encode())
        tx.send_sealed(sealed)

    inf = StreInference(
        inference_id=1,
        pattern_id=PATTERN_ID["supply_convoy"],
        evidence_ids=[1, 2, 3, 4],
        entity_id=42,
        threat_level=THREAT_LEVEL["MEDIUM"],
        target_lat=int(48.158500 * 1e6),
        target_lon=int(37.727000 * 1e6),
        eta_sec=120,
        confidence=80,
        actions=[ACTION["increase_isr"], ACTION["reroute"]],
        pattern_status=0,
        evidence_summary=0x01,
        timestamp=now_ts,
    )
    sealed = sealer.seal(inf.encode())
    tx.send_sealed(sealed)

    time.sleep(0.5)
    listener.stop()
    t.join(timeout=2.0)

    print(f"\n=== Network smoke test ===")
    print(f"Sent: 5 observations + 1 inference over localhost UDP")
    print(f"Received: {len(received)} messages")
    print(f"Listener stats: {listener.stats()}")
    print(f"Transmitter stats: {tx.stats()}")

    assert len(received) == 6, f"expected 6 messages, got {len(received)}"
    obs_msgs = [r for r in received if r[0] == "obs"]
    inf_msgs = [r for r in received if r[0] == "inf"]
    assert len(obs_msgs) == 5
    assert len(inf_msgs) == 1
    # Verify decoded contents survived the wire
    assert obs_msgs[0][1].object_class == OBJECT_CLASS["vehicle_tracked"]
    assert inf_msgs[0][1].pattern_id == PATTERN_ID["supply_convoy"]

    # Test counter gap detection: send a message with a counter that skips ahead
    print("\n--- Counter gap test ---")
    # Force-advance counter by 3 by burning 3 next() calls
    counter_state.next()
    counter_state.next()
    counter_state.next()
    obs2 = StreObservation(
        event_id=100, source_id=0xCA51,
        object_class=OBJECT_CLASS["personnel"],
        sensor_type=SENSOR_TYPE["EO"], confidence=50,
        lat=0, lon=0, alt_m=0, heading=0, speed_mps_x10=0,
        timestamp=now_ts,
    )
    sealed2 = sealer.seal(obs2.encode())

    # Spin listener back up
    listener2 = StreUdpListener(psk, bind_host="127.0.0.1", bind_port=19602)
    # Pre-load last_counter to test gap detection
    listener2._last_counter[0xCA51] = 6   # we sent 6 msgs above
    t2 = threading.Thread(
        target=listener2.serve, daemon=True,
        kwargs={"on_observation": lambda *a: received.append(("obs2",) + a)},
    )
    t2.start()
    time.sleep(0.2)
    tx2 = StreUdpTransmitter("127.0.0.1", 19602)
    tx2.send_sealed(sealed2)
    time.sleep(0.3)
    listener2.stop()
    t2.join(timeout=1.0)

    print(f"Listener2 stats (should show counter gap): "
          f"{listener2.stats()}")
    assert listener2.stats()["by_source"]["0xca51"]["counter_gaps"] >= 3

    print("\n[DONE] all network smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
