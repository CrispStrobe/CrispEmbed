#!/usr/bin/env python3
"""Generic Metal-vs-CPU A/B for one crispembed-cli invocation.

  ab_cmd.py OUT.json [--repeat N] [--metal-env K=V ...] [--cpu-env K=V ...] -- BIN ARGS...

Arms run interleaved, each a separate process of the same binary:
  metal: ARGS + `--gpu-backend metal` + metal-env; stderr must show a Metal device
  cpu:   ARGS + `--gpu-backend cpu`   + cpu-env;   stderr must NOT show one
stdout is compared byte-for-byte; PPM/PGM stdout is also compared per pixel
(max abs, mean abs, PSNR). Text stdout gets a character-level similarity ratio.
"""
import argparse, difflib, hashlib, json, math, os, subprocess, sys, time


def parse_pnm(b):
    if not b[:2] in (b"P5", b"P6"):
        return None
    parts, i = [], 2
    while len(parts) < 3:
        while b[i:i + 1].isspace():
            i += 1
        if b[i:i + 1] == b"#":
            while b[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        parts.append(int(b[i:j]))
        i = j
    return (b[:2], parts[0], parts[1], b[i + 1:])


def run(cmd, env_extra, backend):
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.perf_counter()
    p = subprocess.run(cmd + ["--gpu-backend", backend], capture_output=True, env=env)
    wall = time.perf_counter() - t0
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0:
        raise RuntimeError(f"{backend} rc={p.returncode}: {err[-3000:]}")
    mtl = "MTL0" in err or "ggml_metal" in err
    return p.stdout, wall, err, mtl


def main():
    argv = sys.argv[1:]
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--metal-env", nargs="*", default=[])
    ap.add_argument("--cpu-env", nargs="*", default=[])
    ap.add_argument("--allow-cpu-metal", action="store_true",
                    help="engine creates no Metal device in either arm (CPU-default lane)")
    a = ap.parse_args(argv[:k])
    cmd = argv[k + 1:]
    menv = dict(x.split("=", 1) for x in a.metal_env)
    cenv = dict(x.split("=", 1) for x in a.cpu_env)
    walls = {"metal": [], "cpu": []}
    outs, errs, mtls = {}, {}, {}
    for _ in range(a.repeat):
        for be, env in (("metal", menv), ("cpu", cenv)):
            o, w, e, m = run(cmd, env, be)
            walls[be].append(round(w, 3))
            outs[be], errs[be], mtls[be] = o, e, m
    if not a.allow_cpu_metal:
        if not mtls["metal"]:
            raise RuntimeError("metal arm shows no Metal device")
    if mtls["cpu"]:
        raise RuntimeError("cpu arm initialised Metal")
    mo, co = outs["metal"], outs["cpu"]
    rec = {"cmd": [os.path.basename(cmd[0])] + cmd[1:], "metal_env": menv, "cpu_env": cenv,
           "wall_s": walls, "metal_device_seen": mtls["metal"], "byte_identical": mo == co,
           "sha256": {"metal": hashlib.sha256(mo).hexdigest(), "cpu": hashlib.sha256(co).hexdigest()},
           "date": time.strftime("%Y-%m-%d")}
    pm, pc = parse_pnm(mo), parse_pnm(co)
    if pm and pc:
        if pm[:3] != pc[:3]:
            rec["image"] = {"shape_differs": [pm[:3], pc[:3]]}
        else:
            d = [abs(x - y) for x, y in zip(pm[3], pc[3])]
            mse = sum(x * x for x in d) / len(d)
            rec["image"] = {"w": pm[1], "h": pm[2], "max_abs": max(d), "mean_abs": sum(d) / len(d),
                            "psnr_db": (99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)),
                            "n_diff": sum(1 for x in d if x)}
    else:
        ms, cs = mo.decode("utf-8", "replace"), co.decode("utf-8", "replace")
        rec["stdout"] = {"metal": ms[:6000], "cpu": cs[:6000]}
        rec["char_ratio"] = difflib.SequenceMatcher(None, ms, cs, autojunk=False).ratio()
    rec["stderr_tail"] = {k2: v[-2500:] for k2, v in errs.items()}
    json.dump(rec, open(a.out, "w"), indent=1, ensure_ascii=False)
    s = {"out": os.path.basename(a.out), "identical": rec["byte_identical"], "wall_s": walls}
    if "image" in rec:
        s["image"] = rec["image"]
    if "char_ratio" in rec:
        s["char_ratio"] = round(rec["char_ratio"], 5)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
