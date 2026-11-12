"""CrispSorter consumer check: CrispSorter deps/crispembed-0.17.19 branch head against the CrispEmbed v0.17.19 release, CPU only.

Fresh clone of the app and CrispEmbed at immutable commits, fresh download of
the linux-x86_64 archive verified by the app's own verify_crispembed_bundle.py
against the app's runtime lock, library load, an embedding call, and the
per-model licence flow: the app's licence_consent module (verbatim, built with
crispembed + crispembed-license-api) resolves a restricted model with that
model's own SPDX via resolve_model_accepting; 'all'/'*' are rejected; a
restricted model is refused without explicit acceptance; a tag accepted for one
model does not unlock another. CRISPEMBED_ACCEPT_LICENSE is never set.
Quality: the consented posformer-crohme call must return the expected LaTeX for
a legibly rendered formula, and a permissive printed-text recogniser (parseq)
must read a printed line; both are scored by CER, not mere non-emptiness.
Output: report.json plus one short probe log.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path('/kaggle/temp/pin'); OUT = Path('/kaggle/working/consumer'); OUT.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)
APP = os.environ.get('APP_COMMIT', 'b45dde889edb24bb2feabf80bbfbc24d510df328')
EMBED = 'c364feb974b91e14cbf328c6949f3df36e89ce81'; L = '0.17.19'
report = {'app': APP, 'embed': EMBED, 'checks': [], 'completed': False}
def save(): (OUT / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
LOG = ROOT / 'log.txt'

def run(cmd, cwd=None, env=None, timeout=3600):
    print('RUN', cmd[:3], flush=True); t = time.monotonic()
    with LOG.open('a') as log:
        log.write('\n$ ' + ' '.join(map(str, cmd)) + '\n'); log.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(15)
            if time.monotonic() - t > timeout: p.kill(); raise TimeoutError(str(cmd))
    if p.returncode:
        (OUT / 'log-tail.txt').write_text(LOG.read_text()[-6000:])
        raise RuntimeError(f'{cmd} exited {p.returncode}')

def block(src, pattern):
    m = re.search(pattern, src, re.M); start = src.rfind('\n', 0, m.start()) + 1
    end = re.search(r'^\}\s*$', src[m.end():], re.M)
    return src[start:m.end() + end.end()] + '\n'

def check(name, ok, detail=''):
    report['checks'].append({'name': name, 'pass': bool(ok), 'detail': str(detail)[:240]}); save()
    print('CHECK', name, 'PASS' if ok else 'FAIL', str(detail)[:240], flush=True)

env0 = os.environ.copy()
env0.update(CARGO_HOME=str(ROOT / 'cargo'), RUSTUP_HOME=str(ROOT / 'rustup'), CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0',
            CARGO_TARGET_DIR=str(ROOT / 'target'), CARGO_NET_RETRY='5')
for k in ('CRISPEMBED_ACCEPT_LICENSE', 'CRISPSORTER_ACCEPT_MODEL_LICENSE'): env0.pop(k, None)
save()
try:
    inst = ROOT / 'rustup.sh'; inst.write_bytes(urllib.request.urlopen('https://sh.rustup.rs', timeout=60).read())
    run(['sh', str(inst), '-y', '--profile', 'minimal', '--default-toolchain', '1.98.1', '--no-modify-path'], env=env0)
    env0['PATH'] = str(ROOT / 'cargo/bin') + ':' + env0['PATH']
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import matplotlib; from matplotlib import font_manager
    # Font bundled with matplotlib: no silent fallback to Pillow's 10px default
    # font, which renders an illegible 36x10 px formula (the earlier garbled run).
    fpath = font_manager.findfont('DejaVu Sans', fallback_to_default=False)
    def render(text, size, path, font_px):
        img = Image.new('RGB', size, 'white'); ImageDraw.Draw(img).text((16, 10), text, fill='black', font=ImageFont.truetype(fpath, font_px))
        img.save(path); bb = ImageOps.invert(img.convert('L')).getbbox(); return bb[3] - bb[1]
    fimg, timg = ROOT / 'formula.png', ROOT / 'textline.png'
    ink = (render('x + y = 3', (360, 96), fimg, 48), render('Invoice due 2026', (420, 64), timg, 40))
    check('test images rendered legibly (ink height px)', min(ink) >= 25, ink)
    app, emb = ROOT / 'CrispSorter', ROOT / 'CrispEmbed'
    for name, rev, dst in [('CrispSorter', APP, app), ('CrispEmbed', EMBED, emb)]:
        run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', f'https://github.com/CrispStrobe/{name}.git', str(dst)])
        run(['git', 'checkout', '-q', '--detach', rev], cwd=dst)
    tag = subprocess.run(['git', 'rev-parse', f'v{L}^{{commit}}'], cwd=emb, capture_output=True, text=True).stdout.strip()
    check('tag v0.17.19 is the pinned commit', tag == EMBED, tag)
    lock = json.loads((app / 'scripts/crispembed-runtime-lock.json').read_text())
    check('app runtime lock', lock['version'] == L and lock['source_commit'] == EMBED, lock['version'])
    mlock = json.loads((app / 'scripts/ocr-mobile-runtime-lock.json').read_text())
    check('app mobile lock', mlock['source_commit'] == EMBED, mlock['source_commit'][:12])
    arc = ROOT / 'crispembed-linux-x86_64.tar.gz'
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f'https://github.com/CrispStrobe/CrispEmbed/releases/download/v{L}/{arc.name}', timeout=180) as s, arc.open('wb') as f: shutil.copyfileobj(s, f)
            break
        except Exception as e:
            print('download retry', attempt, e, flush=True); time.sleep(30 * (attempt + 1))
    else: raise RuntimeError('release download failed after retries')
    v = subprocess.run([sys.executable, 'scripts/verify_crispembed_bundle.py', str(arc), '--asset', arc.name, '--source', str(emb)], cwd=app, capture_output=True, text=True)
    check('fresh install hash-verified', v.returncode == 0, (v.stdout + v.stderr).strip()[-200:])
    libs = emb / 'build'; libs.mkdir(); run(['tar', '-xzf', str(arc), '-C', str(libs)])
    probe = ROOT / 'probe'; (probe / 'src').mkdir(parents=True)
    consent = app / 'src-tauri/src/index/license_consent.rs'
    math = (app / 'src-tauri/src/extractors/math_ocr.rs').read_text()
    fn = block(math, r'^pub fn recognize_formula_with_model\b')
    # Since CrispSorter wrapped formula OCR in operation logging, the probe also
    # carries the app's own logged_formula helper and operation_log module.
    logged = block(math, r'^fn logged_formula\b') if 'logged_formula(' in fn else ''
    oplog = app / 'src-tauri/src/operation_log.rs'
    prov = (app / 'src-tauri/src/model_provisioning.rs').read_text()
    report['math_fn_sha'] = hashlib.sha256(fn.encode()).hexdigest()
    # Desktop branch of model_provisioning::resolve_model, checked to be the app's own text.
    shim = ('if Path::new(name).is_file() { return Ok(name.to_owned()); }\n'
            '        crate::index::license_consent::resolve_registry_model(name, auto_download).map_err(|e| format!("{e:#}"))')
    check('desktop provisioning branch unchanged', shim in prov)
    (probe / 'src/lib.rs').write_text(
        'pub mod index { #[path = ' + json.dumps(str(consent)) + '] pub mod license_consent; }\n'
        + ('#[path = ' + json.dumps(str(oplog)) + '] pub mod operation_log;\n'
           'pub fn app_log(_level: &str, _msg: String) {}\n' if logged else '')
        +
        'pub mod model_provisioning { use std::path::Path; pub fn resolve_model(name: &str, auto_download: Option<bool>) -> Result<String, String> {\n        '
        + shim + '\n} }\n'
        'pub mod extractors { pub mod math_ocr { use anyhow::{Result, Context}; use std::path::Path;\n'
        + ('use crate::operation_log::OperationLog; use serde_json::{json, Value};\n' + logged + '\n' if logged else '')
        + fn + '} }\n')
    (probe / 'src/main.rs').write_text(r'''
use probe::{extractors::math_ocr::recognize_formula_with_model, index::license_consent};
use crispembed::CrispEmbed as C;
fn r(x: Result<String, String>) -> serde_json::Value { match x { Ok(p) => serde_json::json!({"ok": p}), Err(e) => serde_json::json!({"err": e}) } }
fn present(n: &str) -> bool { std::fs::read_dir(C::cache_dir()).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().contains(n))).unwrap_or(false) }
fn main() {
    let img = std::env::args().nth(1).unwrap();
    let timg = std::env::args().nth(2).unwrap();
    let mut o = serde_json::Map::new();
    o.insert("env_accept_set".into(), std::env::var("CRISPEMBED_ACCEPT_LICENSE").is_ok().into());
    let path = C::resolve_model("all-MiniLM-L6-v2", Some(true)).expect("resolve embedder");
    let mut m = C::new(&path, 2).expect("load");
    let v: Vec<Vec<f32>> = ["The invoice is due in March.", "Die Rechnung ist im März fällig."].iter().map(|t| m.encode(t)).collect();
    o.insert("dim".into(), m.dim().into());
    o.insert("emb_ok".into(), v.iter().all(|e| e.len() == m.dim() as usize && e.iter().any(|x| *x != 0.0)).into());
    let info = C::model_license_info("posformer-crohme");
    o.insert("native_info".into(), serde_json::json!(info.as_ref().map(|i| (&i.name, &i.spdx, i.requires_acceptance))));
    let app = license_consent::license_info("posformer-crohme");
    o.insert("app_info".into(), serde_json::json!({"spdx": app.spdx, "label": app.label, "requires_consent": app.requires_consent}));
    let spdx = info.map(|i| i.spdx).unwrap_or_default();
    o.insert("accepting_all".into(), r(C::resolve_model_accepting("posformer-crohme", Some(true), "all")));
    o.insert("accepting_star".into(), r(C::resolve_model_accepting("posformer-crohme", Some(true), "*")));
    o.insert("native_plain".into(), r(C::resolve_model("posformer-crohme", Some(true))));
    o.insert("native_wrong_model".into(), r(C::resolve_model_accepting("splade-v3", Some(true), &spdx)));
    o.insert("downloaded_before_consent".into(), (present("posformer") || present("splade")).into());
    let denied = recognize_formula_with_model(std::path::Path::new(&img), "posformer-crohme");
    o.insert("app_without_consent".into(), match &denied { Ok(x) => serde_json::json!({"ok": x}), Err(e) => serde_json::json!({"err": e.to_string()}) });
    o.insert("downloaded_after_refusal".into(), present("posformer").into());
    license_consent::accept("posformer-crohme");
    o.insert("app_other_model".into(), r(license_consent::resolve_registry_model("texo-distill", Some(true)).map_err(|e| e.to_string())));
    let ok = recognize_formula_with_model(std::path::Path::new(&img), "posformer-crohme");
    o.insert("app_with_consent".into(), match &ok { Ok(x) => serde_json::json!({"ok": x}), Err(e) => serde_json::json!({"err": e.to_string()}) });
    // Printed-text OCR with a permissive (apache-2.0) recogniser, no consent needed.
    let tpath = C::resolve_model("parseq", Some(true)).expect("resolve parseq");
    let t = image::open(&timg).expect("open text image").to_rgb8(); let (tw, th) = t.dimensions();
    let mut tm = crispembed::OcrModel::new(&tpath, 0).expect("parseq init");
    o.insert("text_ocr".into(), serde_json::json!(tm.recognize(&t.into_raw(), tw as i32, th as i32)));
    o.insert("env_accept_set_after".into(), std::env::var("CRISPEMBED_ACCEPT_LICENSE").is_ok().into());
    o.insert("other_downloaded".into(), (present("texo") || present("splade")).into());
    println!("PROBE {}", serde_json::Value::Object(o));
}
''')
    (probe / 'Cargo.toml').write_text(f'''[package]
name = "probe"
version = "0.0.0"
edition = "2021"
[workspace]
[features]
default = ["crispembed", "crispembed-license-api"]
crispembed = []
crispembed-license-api = []
[dependencies]
anyhow = "=1.0.102"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
uuid = {{ version = "1", features = ["v4"] }}
image = {{ version = "=0.25.10", default-features = false, features = ["png"] }}
crispembed = {{ path = {json.dumps(str(emb / 'crispembed'))} }}
''')
    env = dict(env0, CRISPEMBED_SYS_LIB_DIR=str(libs), LD_LIBRARY_PATH=str(libs), CRISPEMBED_CACHE_DIR=str(ROOT / 'cache'))
    run(['cargo', 'build', '--release', '--manifest-path', str(probe / 'Cargo.toml')], env=env, timeout=3600)
    check('library build+link (crispembed-license-api)', True)
    p = subprocess.run([str(ROOT / 'target/release/probe'), str(fimg), str(timg)], env=env, capture_output=True, text=True, timeout=1800)
    (OUT / 'probe-tail.txt').write_text((p.stdout + p.stderr)[-4000:])
    line = next((l for l in p.stdout.splitlines() if l.startswith('PROBE ')), None)
    check('probe exit', p.returncode == 0 and line, f'rc={p.returncode}')
    j = json.loads(line[6:]); report['probe'] = j; save()
    E = lambda k: j[k].get('err', '') if isinstance(j[k], dict) else ''
    check('library load + embedding', j['dim'] == 384 and j['emb_ok'], j['dim'])
    ni = j['native_info']
    check('model_license_info: restricted, own SPDX', ni and ni[2] and ni[1].lower() == 'cc-by-nc-sa-3.0', ni)
    check('app licence info matches native SPDX', j['app_info']['spdx'] == (ni[1] if ni else None) and j['app_info']['requires_consent'], j['app_info'])
    check("'all' rejected", 'err' in j['accepting_all'], E('accepting_all'))
    check("'*' rejected", 'err' in j['accepting_star'], E('accepting_star'))
    check('native gate refuses without acceptance', 'err' in j['native_plain'], E('native_plain'))
    check('one model\'s SPDX does not unlock another', 'err' in j['native_wrong_model'], E('native_wrong_model'))
    check('nothing downloaded before consent', not j['downloaded_before_consent'] and not j['downloaded_after_refusal'])
    check('app refuses OCR without consent', 'licen' in E('app_without_consent').lower(), E('app_without_consent'))
    check('consent is per model (texo-distill still refused)', 'err' in j['app_other_model'] and not j['other_downloaded'], E('app_other_model'))
    okv = j['app_with_consent'].get('ok')
    def cer(got, ref):
        a, b = re.sub(r'\s+', '', got or ''), re.sub(r'\s+', '', ref); d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            p, d[0] = d[0], i
            for k, y in enumerate(b, 1): p, d[k] = d[k], min(d[k] + 1, d[k - 1] + 1, p + (x != y))
        return d[len(b)] / len(b)
    # POSFormer is a math recogniser: the input is a formula, expected LaTeX 'x + y = 3'
    # (the released CLI and C API both return it exactly for this image).
    c = cer(okv, 'x + y = 3') if isinstance(okv, str) else 1.0
    check('consented posformer-crohme OCR: formula LaTeX matches (CER<=0.2)', c <= 0.2, f'{okv!r} cer={c:.2f}')
    tv = j.get('text_ocr'); c = cer(tv, 'Invoice due 2026') if isinstance(tv, str) else 1.0
    check('printed-text OCR (parseq, apache-2.0) matches (CER<=0.2, spaces ignored)', c <= 0.2, f'{tv!r} cer={c:.2f}')
    check('CRISPEMBED_ACCEPT_LICENSE never set', not j['env_accept_set'] and not j['env_accept_set_after'])
    report['completed'] = all(c['pass'] for c in report['checks'])
except Exception as e:
    report['error'] = f'{type(e).__name__}: {e}'; raise
finally:
    save(); shutil.rmtree(ROOT / 'target', ignore_errors=True); print(json.dumps(report, indent=1)[:5000])
