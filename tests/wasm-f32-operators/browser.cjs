'use strict';
// Real isolated browser + SIMD/pthread module. No model or complete encoder.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Hosted browser only');
const root=path.resolve(process.argv[2]),browserRoot=path.join(root,'browser');
const report={completed:false,isolated:false,pthreads:false,page_errors:[],outputs:{}};
const save=()=>fs.writeFileSync(path.join(browserRoot,'report.json'),JSON.stringify(report,null,2)+'\n');
const packets=JSON.parse(fs.readFileSync(path.join(browserRoot,'packets.json')));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(browserRoot,'.'+name);
 if(target!==browserRoot&&!target.startsWith(browserRoot+path.sep)){res.writeHead(403).end();return;}
 if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>F32 operator attribution</title>');return;}
 res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
  await page.exposeFunction('retainPacket',(key,row)=>{
   if(!Object.hasOwn(packets,key)||!/^(native|wasm[01])\/[a-z0-9_]+$/.test(key))throw new Error('Unexpected actual packet key');
   const raw=Buffer.from(row.bytes),file='raw/wasm/'+key+'.f32';
   if(hash(raw)!==row.sha256||raw.length!==packets[key].width*12*4)throw new Error('Browser->Node transfer differs');
   fs.mkdirSync(path.dirname(path.join(root,file)),{recursive:true});fs.writeFileSync(path.join(root,file),raw);
   report.outputs[key]={file,bytes:raw.length,sha256:row.sha256,input_hashes:row.input_hashes,
                       shape:[12,packets[key].width],repeat_bit_exact:row.repeat_bit_exact};
   save();
  });
  await page.goto('http://127.0.0.1:'+server.address().port+'/');
  await page.addScriptTag({url:'/module/wasm-f32-operators.js'});
  const result=await page.evaluate(async packets=>{
   if(globalThis.crossOriginIsolated!==true||typeof SharedArrayBuffer!=='function')throw new Error('Isolated SharedArrayBuffer required');
   const m=await window.CrispF32Operator({locateFile:n=>'/module/'+n});
   if(!(m.HEAPU8.buffer instanceof SharedArrayBuffer))throw new Error('Actual pthread shared module required');
   const sha=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b))).map(x=>x.toString(16).padStart(2,'0')).join('');
   const outputs={};
   const call=(op,w,bn,eps,a,b,c,out,threads=1)=>m.ccall('crispembed_f32_operator','number',
      ['number','number','number','number','number','number','number','number','number','number','number'],
      [op,w,12,bn,eps,a,b,c,out,w*12,threads]);
   const input=m._malloc(4608*4),output=m._malloc(4608*4);
   try{
    m.HEAPF32.fill(1,input/4,input/4+4608);
    if(call(0,384,0,1e-12,input,0,0,output)!==4608||!Array.from(m.HEAPF32.subarray(output/4,output/4+4608)).every(x=>x===0))throw new Error('Real norm positive control');
    if(call(0,385,0,1e-12,input,0,0,output)>=0)throw new Error('Wrong-shape rejection');
    m.HEAPF32[input/4]=NaN;
    if(call(0,384,0,1e-12,input,0,0,output)>=0)throw new Error('Nonfinite rejection');
    m.HEAPF32[input/4]=1;
    if(call(0,384,0,1e-12,input,0,0,output)!==4608||!Array.from(m.HEAPF32.subarray(output/4,output/4+4608)).every(x=>x===0))throw new Error('Failure recovery');
   }finally{m._free(input);m._free(output);}
   for(const [key,p]of Object.entries(packets)){
    const pointers={},actualHashes={};let out=0;
    try{
     for(const [label,row]of Object.entries(p.inputs)){
      const bytes=new Uint8Array(await(await fetch('/'+row.file)).arrayBuffer());
      if(bytes.byteLength!==row.bytes||await sha(bytes)!==row.sha256)throw new Error('Authenticated actual input differs: '+key+'/'+label);
      pointers[label]=m._malloc(bytes.byteLength);m.HEAPU8.set(bytes,pointers[label]);actualHashes[label]=await sha(bytes);
     }
     const count=p.width*12;out=m._malloc(count*4);
     const run=()=>call(p.op,p.width,p.b_count,p.epsilon,pointers.a,pointers.b||0,pointers.c||0,out);
     if(run()!==count)throw new Error('Actual operator call failed: '+key);
     const bytes=m.HEAPU8.slice(out,out+count*4);
     if(!Array.from(new Float32Array(bytes.buffer)).every(Number.isFinite))throw new Error('Nonfinite actual output');
     if(run()!==count||await sha(m.HEAPU8.slice(out,out+count*4))!==await sha(bytes))throw new Error('Actual repeat differs');
     for(const [label,row]of Object.entries(p.inputs))
      if(await sha(m.HEAPU8.slice(pointers[label],pointers[label]+row.bytes))!==row.sha256)throw new Error('Operator changed input');
     outputs[key]={bytes:Array.from(bytes),sha256:await sha(bytes),input_hashes:actualHashes,repeat_bit_exact:true};
     await window.retainPacket(key,outputs[key]);
    }finally{if(out)m._free(out);for(const ptr of Object.values(pointers))m._free(ptr);}
   }
   return {isolated:true,pthreads:true,controls:{positive:true,wrong_shape_rejected:true,nonfinite_rejected:true,recovery:true}};
  },packets);
  report.isolated=result.isolated;report.pthreads=result.pthreads;report.controls=result.controls;
  if(Object.keys(report.outputs).length!==Object.keys(packets).length)throw new Error('Actual packet output coverage differs');
  report.completed=true;save();
 }catch(error){report.error=String(error);save();throw error;}
 finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
