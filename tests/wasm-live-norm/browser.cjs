'use strict';
// TEST ONLY. Returned byte buffers are copied before free, including signed zero.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
const {retainVector}=require('../wasm-native-order-norm/vector_bytes.cjs');
const root=path.resolve(process.argv[2]),phase=process.argv[3];
if(!['capture','replay'].includes(phase))throw new Error('phase');
const frozenBytes=fs.readFileSync(path.join(root,'frozen.json')),frozen=JSON.parse(frozenBytes);
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const report={source:frozen.source,completed:false,phase,frozen_sha256:sha(frozenBytes),page_errors:[],cases:{},rows:[]};
const save=()=>fs.writeFileSync(path.join(root,'live-'+phase+'-browser.json'),JSON.stringify(report,null,2));
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(req.url.split('?')[0]);if(name==='/'){res.end('<!doctype html><title>live operands</title>');return;}
 const p=path.resolve(root,'.'+name);if(!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 res.setHeader('Content-Type',p.endsWith('.wasm')?'application/wasm':p.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(p);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 try{
  report.chromium=browser.version();
  if(phase==='capture'){
   if(!JSON.parse(fs.readFileSync(path.join(root,'native-proof.json'))).passed)throw new Error('native byte proof first');
   report.cases.original={};
   for(const gate of ['0','1']){
    report.cases.original[gate]={};
    for(const mode of ['ordinary','observed']){
     const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
     try{
      await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.addScriptTag({url:'/candidate/crispembed_embed.js'});
      const rows=await page.evaluate(async({gate,mode,text})=>{
       if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')throw new Error('isolated pthreads');
       const logs=[],m=await window.CrispEmbedText({locateFile:n=>'/candidate/'+n,printErr:s=>logs.push(s)});
       const set=(k,v)=>{if(m.ccall('wasm_trace_setenv','number',['string','string'],[k,v])!==0)throw new Error('setenv');};
       set('CRISPEMBED_WASM_Q8_0_ROUND','1');set('CRISPEMBED_WASM_NATIVE_ORDER_NORM',gate);set('CRISPEMBED_DEBUG_TOKENS','1');
       set('CRISPEMBED_LIVE_NORM_CAPTURE',mode==='observed'?'/capture':null);
       set('CRISPEMBED_DUMP_LAYERS',mode==='observed'?'1':null);set('CRISPEMBED_DUMP_LAYERS_GGUF',mode==='observed'?'/stages.gguf':null);
       m.FS.mkdir('/capture');m.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
       const ctx=m.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('init');
       const result=[];
       try{
        for(let repeat=0;repeat<2;++repeat){
         const start=logs.length,size=m._malloc(4);let pointer=0;
         try{
          pointer=m.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,text,size]);
          if(!pointer||m.getValue(size,'i32')!==384)throw new Error('shape');
          const row={vector:Array.from(m.HEAPF32.slice(pointer/4,pointer/4+384)),raw_bytes:Array.from(m.HEAPU8.slice(pointer,pointer+1536)),logs:logs.slice(start),files:{}};
          if(mode==='observed'){
           if(m.ccall('crispembed_live_norm_finish','number',['number'],[ctx])!==1)throw new Error('13 connected norms/49 captures required');
           for(const name of m.FS.readdir('/capture').filter(n=>n!=='.'&&n!=='..'))row.files[name]=Array.from(m.FS.readFile('/capture/'+name));
           row.files['stages.gguf']=Array.from(m.FS.readFile('/stages.gguf'));
          }
          result.push(row);
         }finally{if(pointer)m._free(pointer);m._free(size);}
        }
       }finally{m._wasm_embed_free(ctx);}
       return result;
      },{gate,mode,text:frozen.case.text});
      rows.forEach((row,index)=>{
       const prefix=['wasm'+gate,mode,index].join('-'),folder=path.join(root,'arrays',prefix);fs.mkdirSync(folder);
       for(const [name,bytes]of Object.entries(row.files)){if(!/^[a-zA-Z0-9_.]+$/.test(name))throw new Error('unsafe capture');fs.writeFileSync(path.join(folder,name),Buffer.from(bytes));}
       Object.assign(row,retainVector(row.raw_bytes,row.vector));row.vector_file=prefix+'-vector.f32';
       fs.writeFileSync(path.join(root,'arrays',row.vector_file),Buffer.from(row.raw_bytes));row.capture_path=prefix;delete row.files;delete row.raw_bytes;
      });
      report.cases.original[gate][mode]=rows;save();
     }finally{await page.close();}
    }
   }
  }else{
   const packets=JSON.parse(fs.readFileSync(path.join(root,'replay-inputs.json')));
   const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
   try{
    await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.addScriptTag({url:'/component/live-norm-replay.js'});
    const rows=await page.evaluate(async packets=>{
     if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')throw new Error('isolated pthreads');
     const m=await window.LiveNormReplay({locateFile:n=>'/component/'+n,noInitialRun:true}),rows=[];
     for(const packet of packets){
      const bytes=new Uint8Array(await(await fetch('/replay-inputs/'+packet.name+'.f32')).arrayBuffer());if(bytes.length!==18432)throw new Error('shape');
      const x=m._malloc(bytes.length),y=m._malloc(bytes.length);
      try{
       m.HEAPU8.set(bytes,x);
       for(const mode of [0,1])for(const threads of [1,2]){
        if(m._live_norm_replay(mode,x,y,4608,threads)!==4608)throw new Error('actual norm replay');
        const first=Array.from(m.HEAPU8.slice(y,y+bytes.length));
        if(m._live_norm_replay(mode,x,y,4608,threads)!==4608)throw new Error('repeat');
        if(JSON.stringify(first)!==JSON.stringify(Array.from(m.HEAPU8.slice(y,y+bytes.length)))||JSON.stringify(Array.from(bytes))!==JSON.stringify(Array.from(m.HEAPU8.slice(x,x+bytes.length))))throw new Error('replay/input neutrality');
        rows.push({name:packet.name,mode,threads,bytes:first,repeat_exact:true,input_exact:true});
       }
      }finally{m._free(y);m._free(x);}
     }
     return rows;
    },packets);
    for(const row of rows){const file=row.name+'-wasm-'+row.mode+'-'+row.threads+'.f32',data=Buffer.from(row.bytes);fs.writeFileSync(path.join(root,'arrays',file),data);row.file=file;row.bytes=data.length;row.sha256=sha(data);report.rows.push(row);save();}
   }finally{await page.close();}
  }
  if(sha(fs.readFileSync(path.join(root,'frozen.json')))!==report.frozen_sha256)throw new Error('Frozen inputs mutated');
  report.completed=true;save();if(report.page_errors.length)throw new Error('browser page errors');
 }catch(e){report.error=String(e);save();throw e;}
 finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
