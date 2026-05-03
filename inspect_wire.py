"""
inspect_wire.py — decode wire_log.bin into human-readable form.

The wire log is a sequence of length-prefixed sealed CBOR envelopes —
exactly what a tactical radio receiver would see. This tool lets you
inspect those bytes for demo / debugging.

Usage:
    python3 inspect_wire.py path/to/wire_log.bin
    python3 inspect_wire.py path/to/wire_log.bin --psk path/to/key
    python3 inspect_wire.py path/to/wire_log.bin --hex-only      # no decryption

Without a PSK you see envelope headers + ciphertext entropy stats. With
the PSK you see the decoded Type 1 / Type 2 contents.

For demos: the headers alone are interesting because they show the
anti-replay counter advancing monotonically and the entropy-of-ciphertext
matching the spec's >=6.0 bpb requirement.
"""

import argparse
import math
import struct
import sys
from pathlib import Path
from collections import Counter

# Reuse the production decoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stre_codec import (
    CoseEnvelope, StreObservation, StreInference,
    OBJECT_CLASS, PATTERN_ID, ACTION, THREAT_LEVEL, SENSOR_TYPE,
)


def shannon_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = Counter(b)
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def read_wire_log(path: str):
    """Yields raw sealed envelope bytes from a length-prefixed wire log."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(2)
            if len(hdr) < 2:
                return
            (size,) = struct.unpack("!H", hdr)
            payload = f.read(size)
            if len(payload) < size:
                return
            yield payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wire_log", help="Path to wire_log.bin")
    ap.add_argument("--psk", help="Path to file containing 32-byte PSK (binary)")
    ap.add_argument("--hex-only", action="store_true",
                    help="Don't even try to decrypt; show envelope headers only")
    ap.add_argument("--limit", type=int, default=20,
                    help="Max messages to display in detail (default 20)")
    args = ap.parse_args()

    sealer = None
    if args.psk and not args.hex_only:
        psk = Path(args.psk).read_bytes()
        if len(psk) != 32:
            sys.exit(f"PSK must be exactly 32 bytes, got {len(psk)}")
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        aead = ChaCha20Poly1305(psk)

    inv_class = {v: k for k, v in OBJECT_CLASS.items()}
    inv_sensor = {v: k for k, v in SENSOR_TYPE.items()}
    inv_pattern = {v: k for k, v in PATTERN_ID.items()}
    inv_threat = {v: k for k, v in THREAT_LEVEL.items()}
    inv_action = {v: k for k, v in ACTION.items()}

    total = 0
    type1 = 0
    type2 = 0
    total_bytes = 0
    all_ciphertext = b""
    counters_seen = []

    print(f"=== Decoding {args.wire_log} ===\n")

    for i, sealed in enumerate(read_wire_log(args.wire_log)):
        total += 1
        total_bytes += len(sealed)
        env = CoseEnvelope.from_bytes(sealed)
        counters_seen.append(env.counter)
        all_ciphertext += env.ciphertext

        show = i < args.limit
        if show:
            print(f"[msg {i:04d}] sealed={len(sealed):3d}B  "
                  f"src=0x{env.source_id:04x}  ctr={env.counter}  ts={env.timestamp}")

        if args.psk and not args.hex_only:
            try:
                plaintext = aead.decrypt(env.nonce_bytes, env.ciphertext, None)
            except Exception as e:
                if show:
                    print(f"           [decrypt failed: {type(e).__name__}: {e or '(no message)'}]")
                continue
            try:
                import cbor2
                arr = cbor2.loads(plaintext)
                if not isinstance(arr, list) or len(arr) == 0:
                    if show:
                        print(f"           [non-array CBOR payload: {type(arr).__name__} {arr!r}]")
                    continue
                if arr[0] == 0x01:
                    type1 += 1
                    obs = StreObservation.decode(plaintext)
                    if show:
                        cls = inv_class.get(obs.object_class, f"0x{obs.object_class:02x}")
                        sensor = inv_sensor.get(obs.sensor_type, f"0x{obs.sensor_type:02x}")
                        print(f"           Type 1 OBSERVATION  ev={obs.event_id}  "
                              f"class={cls:18s}  sensor={sensor}  "
                              f"conf={obs.confidence}%  "
                              f"lat={obs.lat/1e6:.6f}  lon={obs.lon/1e6:.6f}  "
                              f"hdg={obs.heading}°  speed={obs.speed_mps_x10/10:.1f}m/s")
                elif arr[0] == 0x02:
                    type2 += 1
                    inf = StreInference.decode(plaintext)
                    if show:
                        pname = inv_pattern.get(inf.pattern_id, f"0x{inf.pattern_id:02x}")
                        threat = inv_threat.get(inf.threat_level, f"0x{inf.threat_level:02x}")
                        actions = [inv_action.get(a, hex(a)) for a in inf.actions]
                        print(f"           >> Type 2 INFERENCE  "
                              f"pattern={pname}  threat={threat}  "
                              f"conf={inf.confidence}%  "
                              f"evidence={inf.evidence_ids}  actions={actions}")
                else:
                    if show:
                        print(f"           [unknown msg_type: 0x{arr[0]:02x}, len={len(arr)}]")
            except Exception as e:
                if show:
                    err_type = type(e).__name__
                    err_msg = str(e) or "(no message)"
                    print(f"           [decode failed: {err_type}: {err_msg}]")
                    print(f"           [first 16 bytes of plaintext: {plaintext[:16].hex()}]")

        if i == args.limit:
            print(f"... (suppressing remaining {total} messages, --limit to override)")

    print(f"\n=== Summary ===")
    print(f"  Total messages:    {total}")
    print(f"  Total bytes:       {total_bytes}  ({total_bytes/total:.1f} avg)")
    if args.psk and not args.hex_only:
        print(f"  Type 1 (observation): {type1}")
        print(f"  Type 2 (inference):   {type2}")
    if all_ciphertext:
        ent = shannon_entropy(all_ciphertext)
        print(f"  Wire entropy:      {ent:.2f} bpb (spec target >= 6.0)")
    if counters_seen:
        deltas = [counters_seen[i+1] - counters_seen[i]
                  for i in range(len(counters_seen) - 1)]
        monotonic = all(d > 0 for d in deltas)
        print(f"  Counter monotonic: {'YES ✓' if monotonic else 'NO — replay attack possible'}")
        print(f"  Counter range:     {min(counters_seen)} → {max(counters_seen)}")


if __name__ == "__main__":
    main()
