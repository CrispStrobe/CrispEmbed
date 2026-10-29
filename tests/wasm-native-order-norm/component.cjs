'use strict';
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
const root=path.resolve(process.argv[2]);
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const inputs=JSON.parse(fs.readFileSync(path.join(root,'component-inputs.json')));
const report={completed:false,controls:false,rows:[],page_errors:[]};
const save=()=>fs.writeFileSync(path.join(root,'component-browser.json'),JSON.stringify(report,null,2));
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(req.url.split('?')[0]);
 if(name==='/'){res.end('<!doctype html><title>live norm control</title>');return;}
 const p=path.resolve(root,'.'+name);if(!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 res.setHeader('Content-Type',p.endsWith('.wasm')?'application/wasm':p.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(p);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
  await page.goto('http://127.0.0.1:'+server.address().port+'/');
  await page.addScriptTag({url:'/component/norm-candidate-controls.js'});
  report.results=await page.evaluate(async names=>{
   const m=await window.NativeOrderNormControls({locateFile:n=>'/component/'+n,noInitialRun:true});
   if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')throw new Error('isolated pthread environment required');
   if(m._norm_candidate_controls()!==0)throw new Error('geometry/nonfinite/recovery controls');
   const rows=[];
   for(const name of names){
    const bytes=new Uint8Array(await(await fetch('/component-inputs/'+name+'.f32')).arrayBuffer());
    if(bytes.length!==18432)throw new Error('exact12x384 input');
    const x=m._malloc(bytes.length),y=m._malloc(bytes.length);
    try{
     m.HEAPU8.set(bytes,x);
     if(m._norm_candidate_replay(x,y,4608,1)!==4608)throw new Error('live norm replay');
     const first=Array.from(m.HEAPU8.slice(y,y+bytes.length));
     if(m._norm_candidate_replay(x,y,4608,1)!==4608)throw new Error('live norm repeat');
     const second=Array.from(m.HEAPU8.slice(y,y+bytes.length));
     if(JSON.stringify(first)!==JSON.stringify(second)||JSON.stringify(Array.from(m.HEAPU8.slice(x,x+bytes.length)))!==JSON.stringify(Array.from(bytes)))throw new Error('repeat/input neutrality');
     rows.push({name,bytes:first,repeat_exact:true,input_exact:true});
    }finally{m._free(y);m._free(x);}
   }
   return rows;
  },inputs.map(r=>r.name));
  for(const row of report.results){
   const target=Buffer.from(row.bytes);fs.writeFileSync(path.join(root,'component-inputs',row.name+'.wasm.f32'),target);
   report.rows.push({name:row.name,sha256:sha(target),repeat_exact:row.repeat_exact,input_exact:row.input_exact});
  }
  delete report.results;report.controls=true;report.completed=true;save();
  if(report.page_errors.length)throw new Error('page errors');
 }catch(e){report.error=String(e);save();throw e;}
 finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
