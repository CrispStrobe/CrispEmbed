"""Strict model-hash execution evidence shared by inpainting test drivers."""
import re

SHA256_GATE = "CRISPEMBED_INPAINT_SHA256_ACCEL"
LABELS = frozenset(("migan", "lama", "moebius", "pixelhacker", "sdxl_vae"))
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PROFILE = re.compile(
    r"inpaint_hash: label=(?P<label>[a-z_]+) backend=(?P<backend>sha-ni|scalar) "
    r"requested=(?P<requested>[01]) hardware_available=(?P<hardware_available>[01]) "
    r"bytes=(?P<bytes>[0-9]+) complete_blocks=(?P<complete_blocks>[0-9]+) "
    r"accelerated_blocks=(?P<accelerated_blocks>[0-9]+) sha256=(?P<sha256>[0-9a-f]{64})\s*\Z")


def hash_evidence(stderr, expected, requested=False, require_accel=False, require_models=True):
    """Validate successful loader profiles against independently pinned files.

    ``expected`` maps each required loader label to an exact ``bytes`` count
    and lowercase ``sha256``. Missing historical profiles are allowed only
    when acceleration was not requested, or for an explicitly empty-mask
    bypass with ``require_models=False``. Repeated valid profiles are allowed
    because comparators may initialize several ordinary/observed contexts.
    This proves a verified hashing path; it makes no inference-speed claim.
    """
    if type(requested) is not bool or type(require_accel) is not bool or type(require_models) is not bool:
        raise ValueError("hash evidence flags must be booleans")
    if require_accel and not requested:
        raise ValueError("required SHA acceleration needs an explicit request")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("hash evidence requires independently pinned model identities")
    for label, pin in expected.items():
        if (label not in LABELS or not isinstance(pin, dict) or type(pin.get("bytes")) is not int
                or not 0 < pin["bytes"] <= 4 * 1024**3
                or not isinstance(pin.get("sha256"), str) or HEX64.fullmatch(pin["sha256"]) is None):
            raise ValueError("invalid expected model hash identity")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if not isinstance(stderr, str):
        raise ValueError("hash evidence input must be diagnostic bytes or text")
    profiles = []
    for line in stderr.splitlines():
        if not line.startswith("inpaint_hash:"):
            continue
        match = PROFILE.fullmatch(line)
        if match is None:
            raise ValueError("malformed inpaint_hash execution profile")
        row = match.groupdict()
        for key in ("bytes", "complete_blocks", "accelerated_blocks"):
            row[key] = int(row[key])
        for key in ("requested", "hardware_available"):
            row[key] = row[key] == "1"
        label = row["label"]
        if label not in expected:
            raise ValueError("unexpected inpaint_hash model label")
        pin = expected[label]
        if row["bytes"] != pin["bytes"] or row["sha256"] != pin["sha256"]:
            raise ValueError("inpaint_hash model identity differs from its independent pin")
        if row["requested"] != requested:
            raise ValueError("inpaint_hash request differs from the explicit gate")
        if row["complete_blocks"] != row["bytes"] // 64:
            raise ValueError("inpaint_hash complete-block count differs from the file size")
        selected = requested and row["hardware_available"]
        if row["backend"] != ("sha-ni" if selected else "scalar"):
            raise ValueError("inpaint_hash backend does not match actual requested/available selection")
        if row["accelerated_blocks"] != (row["complete_blocks"] if selected else 0):
            raise ValueError("inpaint_hash accelerated-block count does not prove the selected path")
        if require_accel and (not selected or row["accelerated_blocks"] <= 0):
            raise ValueError("actual SHA-NI execution was required but not demonstrated")
        profiles.append(row)
    observed = {row["label"] for row in profiles}
    covered = set(expected) <= observed
    if requested and require_models and not covered:
        raise ValueError("requested model hash acceleration supplied no complete loader evidence")
    hardware_verified = covered and all(row["backend"] == "sha-ni" and row["accelerated_blocks"] > 0
                                        for row in profiles)
    return {"requested": requested, "require_hardware": require_accel, "profiles": profiles,
            "verified_model_labels": sorted(observed), "all_model_identities_verified": covered,
            "hardware_acceleration_verified": hardware_verified,
            "scalar_fallback": requested and any(row["backend"] == "scalar" for row in profiles),
            "empty_mask_bypass_without_hash_profiles": not require_models and not profiles,
            "scope": "Actual verified model hashing and full input-block counts only; no inference or hardware-speed qualification."}
