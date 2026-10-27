'use strict';
// Diagnostic module only. Actual model and arrays are fetched on hosted CI.
const fs=require('fs'),path=require('path'),http=require('http');
const {chromium}=require('playwright');
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Hosted execution only');
const root=path.resolve(process.argv[2]),phase=process.argv[3];
if(!['capture','replay'].includes(phase))throw new Error('Unknown phase');
const reportPath=path.join(root,'report.json');
const report=phase==='capture'?{state:'running',captures:{},page_errors:[]}:JSON.parse(fs.readFileSync(reportPath));
const weights=JSON.parse(fs.readFileSync(path.join(root,'weights.json')));
const save=()=>fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
const server=http.createServer((req,res)=>{
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');
 res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(root,'.'+name);
 if(target!==root&&!target.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Layer3 diagnostic</title>');return;}
 res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
 const base='http://127.0.0.1:'+server.address().port+'/';
 try{
  async function run(gate,capture,replay){
   const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
   try{
    await page.goto(base);await page.addScriptTag({url:'/diagnostic/crispembed_embed.js'});
    return await page.evaluate(async({gate,capture,replay,weights})=>{
     if(globalThis.crossOriginIsolated!==true||typeof SharedArrayBuffer!=='function')throw new Error('Cross-origin-isolated SharedArrayBuffer required before module init');
     const logs=[],m=await window.CrispEmbedText({locateFile:n=>'/diagnostic/'+n,printErr:s=>logs.push(s)});
     const set=(k,v)=>{if(m.ccall('wasm_trace_setenv','number',['string','string'],[k,v])!==0)throw new Error('setenv');};
     set('CRISPEMBED_WASM_Q8_0_ROUND',gate);set('CRISPEMBED_DEBUG_TOKENS','1');set('CRISPEMBED_EMBED_RAW_NORM','1');
     set('CRISPEMBED_DUMP_LAYERS',capture?'1':null);set('CRISPEMBED_DUMP_LAYERS_GGUF',capture?'/stages.gguf':null);
     set('CRISPEMBED_LAYER3_TRACE',capture?'1':null);set('CRISPEMBED_LAYER3_KERNEL',capture?'/kernel':null);
     if(capture)m.FS.mkdir('/kernel');
     m.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
     const ctx=m.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('init');
     const files={};
     function encode(){
      const size=m._malloc(4);let ptr=0;
      try{
       ptr=m.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,'A small reproducible embedding consumer.',size]);
       if(!ptr||m.getValue(size,'i32')!==384)throw new Error('encode shape');
       const vector=Array.from(m.HEAPF32.subarray(ptr/4,ptr/4+384));if(!vector.every(Number.isFinite))throw new Error('nonfinite');return vector;
      }finally{if(ptr)m._free(ptr);m._free(size);}
     }
     function retainKernel(directory,prefix,selected){
      for(const weight of selected)for(const suffix of ['json','input.f32','converted.bin','weight.bin','dots.f32'])
       files[prefix+'/'+weight+'.'+suffix]=Array.from(m.FS.readFile(directory+'/'+weight+'.'+suffix));
     }
     try{
      if(!replay){
       const vector=encode();
       if(capture){files['stages.gguf']=Array.from(m.FS.readFile('/stages.gguf'));retainKernel('/kernel','kernel',Object.keys(weights));}
       return {vector,logs,files};
      }
      for(const axis of ['native','candidate'])for(const [weight,meta]of Object.entries(weights)){
       const bytes=new Uint8Array(await(await fetch('/arrays/'+axis+'-'+weight+'.input.f32')).arrayBuffer());
       if(bytes.byteLength!==meta.shape[0]*12*4)throw new Error('replay input shape');
       const directory='/replay-'+axis+'-'+weight;m.FS.mkdir(directory);set('CRISPEMBED_LAYER3_KERNEL',directory);
       const input=m._malloc(bytes.byteLength),output=m._malloc(meta.shape[1]*12*4);
       try{
        m.HEAPU8.set(bytes,input);
        const n=m.ccall('crispembed_layer3_replay','number',['number','string','number','number','number','number'],[ctx,weight,input,bytes.byteLength/4,output,meta.shape[1]*12]);
        if(n!==meta.shape[1]*12)throw new Error('replay return '+n);
        files['replay/'+axis+'/'+weight+'.output.f32']=Array.from(m.HEAPU8.subarray(output,output+n*4));
        retainKernel(directory,'replay/'+axis,[weight]);
       }finally{m._free(input);m._free(output);}
      }
      set('CRISPEMBED_LAYER3_KERNEL',null);
      return {vector_after_replay:encode(),logs,files};
     }finally{m._wasm_embed_free(ctx);}
    },{gate,capture,replay,weights});
   }finally{await page.close();}
  }
  function retain(row,prefix){
   for(const [name,bytes]of Object.entries(row.files)){
    const target=path.join(root,'arrays',prefix,name);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,Buffer.from(bytes));
   }
   delete row.files;return row;
  }
  if(phase==='capture'){
   for(const gate of ['0','1']){
    report.captures[gate]={ordinary:retain(await run(gate,false,false),'unused')};save();
    const row=await run(gate,true,false);
    for(const [name,bytes]of Object.entries(row.files)){
     let relative=name==='stages.gguf'?'wasm'+gate+'-stages.gguf':'wasm'+gate+'/'+name.slice('kernel/'.length);
     const target=path.join(root,'arrays',relative);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,Buffer.from(bytes));
     if(gate==='1'&&name.endsWith('.input.f32'))fs.writeFileSync(path.join(root,'arrays','candidate-'+name.slice('kernel/'.length)),Buffer.from(bytes));
    }
    delete row.files;report.captures[gate].capture=row;save();
   }
   report.state='captured';
  }else{
   if(report.state!=='captured')throw new Error('capture required first');
   const anchors=JSON.parse(fs.readFileSync(path.join(root,'anchors.json')));
   if(anchors.passed!==true||anchors.source!==process.env.GITHUB_SHA)throw new Error('anchors required before replay');
   report.replay=retain(await run('1',false,true),'');report.state='complete';
  }
  report.chromium=browser.version();save();if(report.page_errors.length)throw new Error('page errors');
 }catch(error){report.state='failed';report.error=String(error);save();throw error;}
 finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
