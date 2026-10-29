'use strict';
// TEST ONLY; no full encoder, reference-vector injection or production export.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
if(process.env.GITHUB_ACTIONS!=='true'||process.env.RUNNER_ENVIRONMENT!=='github-hosted')throw Error('Genuine hosted browser required');
const root=path.resolve(process.argv[2]),base=path.join(root,'browser'),phase=process.argv[3];
if(!['baseline','candidate'].includes(phase))throw Error('Explicit replay phase required');
if(phase==='candidate'){const a=JSON.parse(fs.readFileSync(path.join(root,'review/cast-anchors.json')));if(a.passed!==true||Object.keys(a.anchors).length!==6)throw Error('Six original anchors required');}
const packets=JSON.parse(fs.readFileSync(path.join(base,'cast-packets.json')));
const report=phase==='candidate'?JSON.parse(fs.readFileSync(path.join(base,'cast-report.json'))):{completed:false,outputs:{},page_errors:[]};
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const save=()=>fs.writeFileSync(path.join(base,'cast-report.json'),JSON.stringify(report,null,2)+'\n');
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(base,'.'+name);
 if(target!==base&&!target.startsWith(base+path.sep)){res.writeHead(403).end();return;}
 if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Layer3 F32 cast risk</title>');return;}
 res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
  await page.exposeFunction('retainCast',(key,row)=>{
   if(!/^(quant|cast)_(native|candidate)$/.test(key))throw Error('Unexpected arm');
   const raw=Buffer.from(row.bytes);if(raw.length!==18432||hash(raw)!==row.sha256)throw Error('Output transfer changed');
   const file='raw/wasm/'+key+'.f32';fs.mkdirSync(path.join(root,'raw/wasm'),{recursive:true});fs.writeFileSync(path.join(root,file),raw);
   if(row.coefficients){const c=Buffer.from(row.coefficients);if(c.length!==2359296||hash(c)!==packets.coefficient_sha256)throw Error('Derived coefficients differ from independent decoder');fs.writeFileSync(path.join(root,'raw/wasm/coefficients.f32'),c);}
   const {bytes,coefficients,...metadata}=row;report.outputs[key]={file,bytes:raw.length,...metadata};save();
  });
  await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.addScriptTag({url:'/module/wasm-layer3-f32.js'});
  Object.assign(report,await page.evaluate(async({packets,phase})=>{
   if(globalThis.crossOriginIsolated!==true||typeof SharedArrayBuffer!=='function')throw Error('Isolated shared memory required');
   const m=await window.CrispLayer3F32({locateFile:n=>'/module/'+n});if(!(m.HEAPU8.buffer instanceof SharedArrayBuffer))throw Error('Actual pthread module required');
   const sha=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b))).map(x=>x.toString(16).padStart(2,'0')).join('');
   const load=async r=>{const b=new Uint8Array(await(await fetch('/'+r.file)).arrayBuffer());if(b.length!==r.bytes||await sha(b)!==r.sha256)throw Error('Input changed');return b;};
   const weight=await load(packets.weight),w=m._malloc(weight.length),x=m._malloc(73728),y=m._malloc(18432),d=m._malloc(2359296);
   const call=(mode,coef=0,wb=313344,nx=18432,nd=589824)=>m.ccall('crispembed_layer3_f32','number',Array(10).fill('number'),[mode,w,wb,x,nx,y,4608,coef,coef?nd:0,1]);
   try{
    m.HEAPU8.set(weight,w);
    for(const axis of ['native','candidate']){
     const input=await load(packets.inputs[axis]);m.HEAPU8.set(input,x);const mode=phase==='candidate'?1:0,key=(mode?'cast_':'quant_')+axis;
     if(call(mode)!==4608)throw Error('Ordinary graph failed');const bytes=m.HEAPU8.slice(y,y+18432),outputSHA=await sha(bytes);
     if(!Array.from(new Float32Array(bytes.buffer)).every(Number.isFinite))throw Error('Nonfinite result');
     if(call(mode,mode?d:0)!==4608||await sha(m.HEAPU8.slice(y,y+18432))!==outputSHA)throw Error('Observation changed output');
     let derived=null;if(mode){derived=m.HEAPU8.slice(d,d+2359296);if(await sha(derived)!==packets.coefficient_sha256)throw Error('Actual coefficient cast differs');}
     if(call(mode)!==4608||await sha(m.HEAPU8.slice(y,y+18432))!==outputSHA)throw Error('Repeat differs');
     if(await sha(m.HEAPU8.slice(w,w+weight.length))!==packets.weight.sha256||await sha(m.HEAPU8.slice(x,x+input.length))!==packets.inputs[axis].sha256)throw Error('Input mutated');
     if(!mode&&outputSHA!==packets.targets.wasm[axis].sha256)throw Error('Old browser quantized anchor changed');
     await window.retainCast(key,{bytes:Array.from(bytes),sha256:outputSHA,input_sha256:packets.inputs[axis].sha256,coefficients:derived?Array.from(derived):null,coefficient_sha256:derived?packets.coefficient_sha256:null,ordinary_observed_bit_exact:true,repeat_bit_exact:true,inputs_unchanged:true});
    }
    if(phase==='candidate'){
    m.HEAPU8.fill(0,x,x+73728);m.HEAPF32.fill(17,y/4,y/4+4608);m.HEAPF32.fill(17,d/4,d/4+589824);
    const ys=await sha(m.HEAPU8.slice(y,y+18432)),ds=await sha(m.HEAPU8.slice(d,d+2359296));
    const rejected=async f=>{if(f()>=0||await sha(m.HEAPU8.slice(y,y+18432))!==ys||await sha(m.HEAPU8.slice(d,d+2359296))!==ds)throw Error('Failure output/coefficients changed');};
    await rejected(()=>call(3,d));await rejected(()=>call(1,d,313343));await rejected(()=>call(1,d,313344,18431));await rejected(()=>call(1,d,313344,18432,589823));
    m.HEAPF32[x/4]=NaN;await rejected(()=>call(1,d));m.HEAPF32[x/4]=0;
    const a=m.HEAPU8[w],b=m.HEAPU8[w+1];m.HEAPU8[w]=0;m.HEAPU8[w+1]=124;await rejected(()=>call(1,d));m.HEAPU8[w]=a;m.HEAPU8[w+1]=b;
    if(call(1,d)!==4608||!Array.from(m.HEAPF32.subarray(y/4,y/4+4608)).every(v=>v===0)||await sha(m.HEAPU8.slice(d,d+2359296))!==packets.coefficient_sha256)throw Error('Recovery failed');
    }
   }finally{for(const p of [w,x,y,d])m._free(p);}
   return {isolated:true,pthreads:true,controls:phase==='candidate'?{positive:true,wrong_mode:true,wrong_extents:true,nonfinite_input:true,nonfinite_weight:true,output_preserved:true,coefficient_output_preserved:true,recovery:true}:null};
  },{packets,phase}));
  if(report.page_errors.length||Object.keys(report.outputs).length!==(phase==='candidate'?4:2))throw Error('Incomplete browser phase');
  report.baseline_completed=true;report.completed=phase==='candidate';save();
 }catch(e){report.error=String(e);save();throw e;}finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
