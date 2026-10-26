'use strict';
// Actual packaged modules only; serve them unchanged and use their public wrappers.
const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const { chromium } = require('playwright');
const root = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
function check(ok, message) { if (!ok) throw new Error(message); }
const fixture = JSON.parse(fs.readFileSync(path.join(root, 'native.json'), 'utf8'));
check(fixture.model_sha256 === '77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2', 'Unreviewed model identity');
check(crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'model.gguf'))).digest('hex') === fixture.model_sha256, 'Model bytes differ');
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const target = path.resolve(root, '.' + name);
  if (target !== root && !target.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Packaged module consumer</title>'); return; }
  res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : name.endsWith('.js') ? 'application/javascript' : 'application/octet-stream');
  const stream = fs.createReadStream(target); stream.on('error', () => { res.writeHead(404).end(); }); stream.pipe(res);
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true});
  const errors = [];
  try {
    const page = await browser.newPage(); page.on('pageerror', e => errors.push(String(e)));
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    await page.addScriptTag({url: '/crispembed-embed-wasm/crispembed_embed.js'});
    const embed = await page.evaluate(async (text) => {
      const module = await window.CrispEmbedText({locateFile: name => '/crispembed-embed-wasm/' + name});
      const expected = ['_wasm_embed_init','_wasm_embed_dim','_wasm_embed_set_prefix','_wasm_embed_encode_copy','_wasm_embed_encode_batch_copy','_wasm_embed_free','_malloc','_free'];
      for (const name of expected) if (typeof module[name] !== 'function') throw new Error('Missing packaged export ' + name);
      module.FS.writeFile('/model.gguf', new Uint8Array(await (await fetch('/model.gguf')).arrayBuffer()));
      const ctx = module.ccall('wasm_embed_init', 'number', ['string','number'], ['/model.gguf',1]);
      if (!ctx) throw new Error('Actual packaged model init failed');
      const size = module._malloc(4);
      try {
        const ptr = module.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,text,size]);
        const dim = module.getValue(size,'i32');
        if (!ptr || dim !== 384 || module._wasm_embed_dim(ctx) !== 384) throw new Error('Embedding shape failed');
        const vector = Array.from(module.HEAPF32.subarray(ptr/4,ptr/4+dim)); module._free(ptr);
        return {exports: expected, vector};
      } finally { module._free(size); module._wasm_embed_free(ctx); }
    }, fixture.text);
    await page.addScriptTag({url: '/crispembed-ocr-wasm/crispembed_ocr.js'});
    const ocr = await page.evaluate(async () => {
      const module = await window.CrispEmbedOCR({locateFile: name => '/crispembed-ocr-wasm/' + name});
      const expected = ['_wasm_ocr_init','_wasm_ocr_recognize','_wasm_ocr_free','_wasm_scan_cleanup_init','_wasm_scan_cleanup_process','_wasm_scan_cleanup_free_image','_wasm_scan_cleanup_free'];
      for (const name of expected) if (typeof module[name] !== 'function') throw new Error('Missing OCR export ' + name);
      const ctx = module.ccall('wasm_scan_cleanup_init','number',['string','number'],['',1]);
      if (!ctx) throw new Error('Classical cleanup init failed');
      const input = new Uint8Array(64*64*3).fill(255);
      for (let y=20;y<44;y++) for(let x=15;x<49;x++) input.fill(0,3*(64*y+x),3*(64*y+x)+3);
      const ptr = module._malloc(input.length); const dims = module._malloc(8); module.HEAPU8.set(input,ptr);
      let result=0;
      try {
        result=module.ccall('wasm_scan_cleanup_process','number',Array(11).fill('number'),[ctx,ptr,64,64,3,0,0,0,0,dims,dims+4]);
        if (!result || module.getValue(dims,'i32')!==64 || module.getValue(dims+4,'i32')!==64) throw new Error('Cleanup dimensions failed');
        const actual = module.HEAPU8.subarray(result,result+input.length);
        if (actual.some((v,i)=>v!==input[i])) throw new Error('No-op cleanup changed pixels');
        return {exports:expected, classical_noop_rgb_exact:true, size:[64,64]};
      } finally { if(result) module._wasm_scan_cleanup_free_image(result); module._free(ptr); module._free(dims); module._wasm_scan_cleanup_free(ctx); }
    });
    check(embed.vector.length===384 && embed.vector.every(Number.isFinite), 'Nonfinite browser embedding');
    const a=embed.vector,b=fixture.vector;
    const maxAbs=Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
    const aa=a.reduce((s,v)=>s+v*v,0),bb=b.reduce((s,v)=>s+v*v,0);
    const cosine=a.reduce((s,v,i)=>s+v*b[i],0)/Math.sqrt(aa*bb);
    const report={schema:'crispembed-successor-wasm-consumer-v1',source_commit:process.env.GITHUB_SHA,model_sha256:fixture.model_sha256,embed,ocr,max_abs:maxAbs,cosine,page_errors:errors,
      limits:['One published MiniLM model and classical cleanup; no OCR model-quality or browser performance claim.']};
    fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
    check(errors.length===0 && maxAbs<=.002 && cosine>=.99999,'Packaged browser/native parity failed');
  } finally { await browser.close(); await new Promise(resolve=>server.close(resolve)); }
})().catch(error=>{ console.error(error);process.exitCode=1; });
