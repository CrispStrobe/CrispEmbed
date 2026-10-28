'use strict';
// Test-only real browser graph admission. No complete encoder or model forward.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
if(process.env.GITHUB_ACTIONS!=='true'||process.env.RUNNER_ENVIRONMENT!=='github-hosted')throw new Error('Genuine hosted browser required');
const root=path.resolve(process.argv[2]),browserRoot=path.join(root,'browser');
const packets=JSON.parse(fs.readFileSync(path.join(browserRoot,'q8k-packets.json')));
const phase=process.argv[3]||'baseline';if(!['baseline','interventions'].includes(phase))throw new Error('Explicit diagnostic phase required');
if(phase==='interventions'){const proof=JSON.parse(fs.readFileSync(path.join(root,'review/cross-anchors.json')));if(proof.passed!==true||Object.keys(proof.anchors).length!==18)throw new Error('Eighteen baselines required before hybrids');}
const report=phase==='interventions'?JSON.parse(fs.readFileSync(path.join(browserRoot,'q8k-report.json'))):{completed:false,baseline_completed:false,isolated:false,pthreads:false,page_errors:[],outputs:{}};
const selected=Object.keys(packets).filter(key=>key.includes('_scale_')===(phase==='interventions'));
const save=()=>fs.writeFileSync(path.join(browserRoot,'q8k-report.json'),JSON.stringify(report,null,2)+'\n');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(browserRoot,'.'+name);
 if(target!==browserRoot&&!target.startsWith(browserRoot+path.sep)){res.writeHead(403).end();return;}
 if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Q8_K cross-admission</title>');return;}
 res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
  await page.exposeFunction('retainQ8K',(key,row)=>{
   if(!Object.hasOwn(packets,key)||!/^[a-z0-9_]+$/.test(key))throw new Error('Unexpected arm');
   const raw=Buffer.from(row.bytes),file='raw/wasm/'+key+'/output.f32';
   if(raw.length!==18432||hash(raw)!==row.sha256)throw new Error('Output transfer changed');
   const folder=path.dirname(path.join(root,file));fs.mkdirSync(folder,{recursive:true});fs.writeFileSync(path.join(root,file),raw);
   if(row.converted){const q=Buffer.from(row.converted);if(q.length!==21024||hash(q)!==row.workspace_sha256)throw new Error('Actual conversion transfer changed');fs.writeFileSync(path.join(folder,'converted.bin'),q);}
   report.outputs[key]={file,bytes:raw.length,sha256:row.sha256,workspace_sha256:row.workspace_sha256,input_sha256:row.input_sha256,repeat_bit_exact:true};save();
  });
  await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.addScriptTag({url:'/module/wasm-q8k-cross.js'});
  const status=await page.evaluate(async ({packets,selected})=>{
   if(globalThis.crossOriginIsolated!==true||typeof SharedArrayBuffer!=='function')throw new Error('Isolated SharedArrayBuffer required');
   const m=await window.CrispQ8KCross({locateFile:n=>'/module/'+n});
   if(!(m.HEAPU8.buffer instanceof SharedArrayBuffer))throw new Error('Actual pthread module required');
   const sha=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b))).map(x=>x.toString(16).padStart(2,'0')).join('');
   const load=async row=>{const bytes=new Uint8Array(await(await fetch('/'+row.file)).arrayBuffer());if(bytes.length!==row.bytes||await sha(bytes)!==row.sha256)throw new Error('Input payload changed');return bytes;};
   const setdir=dir=>{if(m.ccall('crispembed_q8k_setenv','number',[dir?'string':'number'],[dir||0])!==0)throw new Error('Capture setting failed');};
   const call=(mode,w,x,q,out,wb=313344,qb=21024)=>m.ccall('crispembed_q8k_cross','number',Array(10).fill('number'),[mode,w,wb,x,18432,q,mode?qb:0,out,4608,1]);
   for(const key of selected){const p=packets[key];
    const pointers={},inputs={};let output=0;
    try{
     for(const name of ['weight','input','workspace'])if(p[name]){inputs[name]=await load(p[name]);pointers[name]=m._malloc(inputs[name].length);m.HEAPU8.set(inputs[name],pointers[name]);}
     output=m._malloc(18432);const directory='/q8k-'+key;
     if(p.mode===0){m.FS.mkdir(directory);setdir(directory);}else setdir(null);
     const run=()=>call(p.mode,pointers.weight,pointers.input,pointers.workspace||0,output);
     if(run()!==4608)throw new Error('Actual graph call failed:'+key);const bytes=m.HEAPU8.slice(output,output+18432);
     if(!Array.from(new Float32Array(bytes.buffer)).every(Number.isFinite))throw new Error('Nonfinite output');
     if(run()!==4608||await sha(m.HEAPU8.slice(output,output+18432))!==await sha(bytes))throw new Error('Repeat differs');
     for(const [name,v]of Object.entries(inputs))if(await sha(m.HEAPU8.slice(pointers[name],pointers[name]+v.length))!==await sha(v))throw new Error('Read-only input changed');
     let converted=null,workspace=p.workspace?p.workspace.sha256:p.expected_workspace_sha256;
     if(p.mode===0){converted=Array.from(m.FS.readFile(directory+'/blk.3.ffn_down.weight.converted.bin'));if(await sha(new Uint8Array(converted))!==workspace)throw new Error('Actual selected conversion workspace changed');}
     if(p.expected_outputs.wasm&&await sha(bytes)!==p.expected_outputs.wasm.sha256)throw new Error('Normal/direct route anchor failed:'+key);
     await window.retainQ8K(key,{bytes:Array.from(bytes),sha256:await sha(bytes),converted,workspace_sha256:workspace,input_sha256:p.input.sha256});
    }finally{if(output)m._free(output);for(const ptr of Object.values(pointers))m._free(ptr);}
   }
   setdir(null);
   // Real local rejection, unchanged-output and recovery controls.
   const wb=await load(Object.values(packets)[0].weight),w=m._malloc(wb.length),x=m._malloc(73728),q=m._malloc(21024),out=m._malloc(18432);
   try{
    m.HEAPU8.set(wb,w);m.HEAPU8.fill(0,x,x+73728);m.HEAPU8.fill(0,q,q+21024);m.HEAPF32.fill(17,out/4,out/4+4608);
    const preserved=()=>Array.from(m.HEAPF32.subarray(out/4,out/4+4608)).every(v=>v===17);
    if(call(0,w,x,0,out,313343)>=0||!preserved())throw new Error('Wrong extent rejection');
    m.HEAPF32[x/4]=NaN;if(call(0,w,x,0,out)>=0||!preserved())throw new Error('Nonfinite input rejection');m.HEAPF32[x/4]=0;
    m.HEAPF32[q/4]=NaN;if(call(1,w,x,q,out)>=0||!preserved())throw new Error('Nonfinite Q8_K scale rejection');m.HEAPF32[q/4]=0;
    m.HEAPU8[q+260]=1;if(call(1,w,x,q,out)>=0||!preserved())throw new Error('Inconsistent sum rejection');m.HEAPU8[q+260]=0;
    if(call(1,w,x,q,out)!==4608||!Array.from(m.HEAPF32.subarray(out/4,out/4+4608)).every(v=>v===0))throw new Error('Actual recovery failed');
   }finally{for(const ptr of [w,x,q,out])m._free(ptr);}
   return {isolated:true,pthreads:true,controls:{positive:true,wrong_extent_rejected:true,nonfinite_rejected:true,invalid_workspace_rejected:true,output_preserved_on_failure:true,recovery:true}};
  },{packets,selected});
  Object.assign(report,status);const count=phase==='baseline'?6:8;if(Object.keys(report.outputs).length!==count)throw new Error('Complete declared phase coverage required');report.baseline_completed=true;report.completed=phase==='interventions';save();
 }catch(error){report.error=String(error);save();throw error;}
 finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
