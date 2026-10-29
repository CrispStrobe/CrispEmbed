'use strict';
// Actual public WASM C ABI; held Q8=1, independently snapshotted norm gate.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
const root=path.resolve(process.argv[2]),phase=process.argv[3];
if(!['baseline','candidate'].includes(phase))throw new Error('phase');
const frozen=fs.readFileSync(path.join(root,'frozen-panel.json'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const panel=JSON.parse(frozen), report={completed:false,phase,source:panel.source,frozen_sha256:sha(frozen),cases:{},page_errors:[]};
const save=()=>fs.writeFileSync(path.join(root,phase+'-browser.json'),JSON.stringify(report,null,2));
if(phase==='candidate'){
 const proof=JSON.parse(fs.readFileSync(path.join(root,'off-proof.json')));
 if(!proof.passed||proof.frozen_sha256!==sha(frozen))throw new Error('All OFF bytes must pass first');
 report.off_proof_sha256=sha(fs.readFileSync(path.join(root,'off-proof.json')));
}
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(req.url.split('?')[0]);if(name==='/'){res.end('<!doctype html><title>norm candidate</title>');return;}
 const p=path.resolve(root,'.'+name);if(!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 res.setHeader('Content-Type',p.endsWith('.wasm')?'application/wasm':p.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(p);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 try{
  report.chromium=browser.version();
  const arms=phase==='baseline'?['legacy','unset','empty','0']:['1'];
  async function evaluate(item,arm,observed){
   const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
   try{
    await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.addScriptTag({url:'/candidate/crispembed_embed.js'});
    return await page.evaluate(async({item,arm,observed})=>{
     if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')throw new Error('isolated pthreads required');
     const logs=[],m=await window.CrispEmbedText({locateFile:n=>'/candidate/'+n,printErr:s=>logs.push(s)});
     const set=(k,v)=>{if(m.ccall('wasm_trace_setenv','number',['string','string'],[k,v])!==0)throw new Error('env');};
     set('CRISPEMBED_WASM_Q8_0_ROUND',arm==='legacy'?'0':'1');
     set('CRISPEMBED_WASM_NATIVE_ORDER_NORM',arm==='unset'?null:arm==='empty'?'':arm==='legacy'?'0':arm);
     set('CRISPEMBED_DEBUG_TOKENS','1');set('CRISPEMBED_EMBED_RAW_NORM','1');
     set('CRISPEMBED_DUMP_LAYERS',observed?'1':null);
     set('CRISPEMBED_DUMP_LAYERS_GGUF',observed&&item.id==='original'?'/stages.gguf':null);
     set('CRISPEMBED_TRACE_POOL',observed?'/pool':null);
     m.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
     const ctx=m.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('init');
     const rows=[];
     try{
      for(let repeat=0;repeat<2;++repeat){
       const start=logs.length,size=m._malloc(4);let pointer=0;
       try{
        pointer=m.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,item.text,size]);
        if(!pointer||m.getValue(size,'i32')!==384)throw new Error('shape');
        const vector=Array.from(m.HEAPF32.slice(pointer/4,pointer/4+384));if(!vector.every(Number.isFinite))throw new Error('finite');
        const files={};if(observed){
         for(const n of ['pool.mean.f32','pool.final.f32'])files[n]=Array.from(m.FS.readFile('/'+n));
         if(item.id==='original')files['stages.gguf']=Array.from(m.FS.readFile('/stages.gguf'));
        }
        rows.push({vector,logs:logs.slice(start),files});
       }finally{if(pointer)m._free(pointer);m._free(size);}
      }
     }finally{m._wasm_embed_free(ctx);}
     return rows;
    },{item:{id:item.id,text:item.text},arm,observed});
   }finally{await page.close();}
  }
  fs.mkdirSync(path.join(root,'arrays'),{recursive:true});
  for(const item of panel.cases){
   report.cases[item.id]={};
   for(const arm of arms){
    report.cases[item.id][arm]={};
    for(const mode of ['ordinary','observed']){
     const rows=await evaluate(item,arm,mode==='observed');
     rows.forEach((r,i)=>{
      for(const [name,bytes]of Object.entries(r.files))fs.writeFileSync(path.join(root,'arrays',[arm,item.id,mode,i,name].join('-')),Buffer.from(bytes));
      delete r.files;
     });
     report.cases[item.id][arm][mode]=rows;save();
    }
   }
  }
  if(phase==='candidate'){
   const page=await browser.newPage();await page.goto('http://127.0.0.1:'+server.address().port+'/');
   await page.addScriptTag({url:'/candidate/crispembed_embed.js'});
   report.interleaving=await page.evaluate(async text=>{
    const m=await window.CrispEmbedText({locateFile:n=>'/candidate/'+n});
    const set=(k,v)=>m.ccall('wasm_trace_setenv','number',['string','string'],[k,v]);
    set('CRISPEMBED_WASM_Q8_0_ROUND','1');
    set('CRISPEMBED_DEBUG_TOKENS',null);set('CRISPEMBED_DUMP_LAYERS',null);set('CRISPEMBED_DUMP_LAYERS_GGUF',null);set('CRISPEMBED_TRACE_POOL',null);
    m.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
    const contexts=[],rows=[];
    try{
     for(const gate of ['0','1','0']){set('CRISPEMBED_WASM_NATIVE_ORDER_NORM',gate);const ctx=m.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('context');contexts.push(ctx);}
     set('CRISPEMBED_WASM_NATIVE_ORDER_NORM','1');
     for(const index of [0,1,2,0]){
      const size=m._malloc(4);let p=0;try{p=m.ccall('wasm_embed_encode_copy','number',['number','string','number'],[contexts[index],text,size]);if(!p||m.getValue(size,'i32')!==384)throw new Error('context encode');rows.push({index,vector:Array.from(m.HEAPF32.slice(p/4,p/4+384))});}finally{if(p)m._free(p);m._free(size);}
     }
    }finally{contexts.forEach(c=>m._wasm_embed_free(c));}
    return rows;
   },panel.cases[0].text);await page.close();
  }
  if(sha(fs.readFileSync(path.join(root,'frozen-panel.json')))!==report.frozen_sha256)throw new Error('Freeze mutated');
  report.completed=true;save();if(report.page_errors.length)throw new Error('page errors');
 }catch(e){report.error=String(e);save();throw e;}
 finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
