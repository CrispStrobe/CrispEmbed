'use strict';
// Supplementary outputs freeze before candidate inference; all modules stay separate.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require('playwright');
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Hosted execution only');
const root=path.resolve(process.argv[2]),phase=process.argv[3];
if(!['reference','candidate'].includes(phase))throw new Error('Unknown phase');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const cases=JSON.parse(fs.readFileSync(path.join(root,'panel.json')));
const output=path.join(root,phase+'-browser.json');
const report={schema:'crispembed-wasm-q8-ab-browser-v1',phase,cases:{},page_errors:[],state:'running'};
if(phase==='candidate'){
  const bytes=fs.readFileSync(path.join(root,'old-panel-freeze.json'));
  const expected=fs.readFileSync(path.join(root,'old-panel-freeze.sha256'),'utf8').trim();
  if(hash(bytes)!==expected||JSON.parse(bytes).state!=='old-panel-frozen')throw new Error('Immutable old panel freeze required first');
  report.old_panel_freeze_sha256=expected;
}
const save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
const server=http.createServer((req,res)=>{
 const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(root,'.'+name);
 if(target!==root&&!target.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Conditional rounding A/B</title>');return;}
 res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
 const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});report.chromium=browser.version();
 try{
  const base='http://127.0.0.1:'+server.address().port+'/';
  async function evaluate(directory,item,gate,dump){
   const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
   try{
    await page.goto(base);await page.addScriptTag({url:'/'+directory+'/crispembed_embed.js'});
    return await page.evaluate(async({directory,item,gate,dump})=>{
     const logs=[],module=await window.CrispEmbedText({locateFile:name=>'/'+directory+'/'+name,printErr:line=>logs.push(line)});
     if(directory!=='crispembed-embed-wasm'){
      const set=(key,value)=>{if(module.ccall('wasm_trace_setenv','number',['string','string'],[key,value])!==0)throw new Error('setenv failed');};
      set('CRISPEMBED_WASM_Q8_0_ROUND',gate==='unset'?null:gate==='empty'?'':gate);
      set('CRISPEMBED_DEBUG_TOKENS','1');set('CRISPEMBED_EMBED_RAW_NORM','1');
      set('CRISPEMBED_DUMP_LAYERS',dump?'1':null);
      set('CRISPEMBED_DUMP_LAYERS_GGUF',dump&&item.id==='original'?'/stages.gguf':null);
      set('CRISPEMBED_TRACE_POOL',dump?'/pool':null);
     }
     module.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
     const ctx=module.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('model init');
     const size=module._malloc(4);
     try{
      const ptr=module.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,item.text,size]);const n=module.getValue(size,'i32');
      if(!ptr||n!==384)throw new Error('encode shape');const vector=Array.from(module.HEAPF32.subarray(ptr/4,ptr/4+n));module._free(ptr);
      if(!vector.every(Number.isFinite))throw new Error('nonfinite vector');
      const files={};if(dump){
       for(const name of ['pool.mean.f32','pool.final.f32'])files[name]=Array.from(module.FS.readFile('/'+name));
       if(item.id==='original')files['stages.gguf']=Array.from(module.FS.readFile('/stages.gguf'));
      }
      return {vector,logs,files,norm:Math.sqrt(vector.reduce((s,x)=>s+x*x,0))};
     }finally{module._free(size);module._wasm_embed_free(ctx);}
    },{directory,item,gate,dump});
   }finally{await page.close();}
  }
  function retain(name,result){
   for(const [file,bytes]of Object.entries(result.files))fs.writeFileSync(path.join(root,'arrays',name+'-'+file),Buffer.from(bytes));
   delete result.files;return result;
  }
  fs.mkdirSync(path.join(root,'arrays'),{recursive:true});
  for(const item of cases){
   report.cases[item.id]={};
   if(phase==='reference'){
    report.cases[item.id].packaged=retain('old-packaged-'+item.id,await evaluate('crispembed-embed-wasm',item,'unset',false));save();
    report.cases[item.id].diagnostic=retain('old-diagnostic-'+item.id,await evaluate('old-diagnostic',item,'unset',false));save();
    if(item.id==='original')report.cases[item.id].dump=retain('old-dump-'+item.id,await evaluate('old-diagnostic',item,'unset',true));
   }else{
    for(const gate of ['unset','empty','0','1']){
     report.cases[item.id][gate]={};
     for(const mode of ['ordinary','dump']){
      report.cases[item.id][gate][mode]=retain('candidate-'+gate+'-'+item.id,await evaluate('candidate',item,gate,mode==='dump'));save();
     }
    }
   }
   save();
  }
  if(phase==='candidate'){
   const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
   try{
    await page.goto(base);await page.addScriptTag({url:'/candidate/crispembed_embed.js'});
    report.interleaving=await page.evaluate(async(text)=>{
     const logs=[],m=await window.CrispEmbedText({locateFile:name=>'/candidate/'+name,printErr:line=>logs.push(line)});
     const set=(k,v)=>{if(m.ccall('wasm_trace_setenv','number',['string','string'],[k,v])!==0)throw new Error('setenv');};
     set('CRISPEMBED_DEBUG_TOKENS','1');set('CRISPEMBED_EMBED_RAW_NORM','1');
     set('CRISPEMBED_DUMP_LAYERS',null);set('CRISPEMBED_DUMP_LAYERS_GGUF',null);set('CRISPEMBED_TRACE_POOL',null);
     m.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
     const contexts=[];
     try{
      for(const flag of ['0','1','0']){set('CRISPEMBED_WASM_Q8_0_ROUND',flag);const ctx=m.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);if(!ctx)throw new Error('interleaved init');contexts.push(ctx);}
      // Leave the process environment at1: existing0 contexts must stay0.
      set('CRISPEMBED_WASM_Q8_0_ROUND','1');const rows=[];
      for(const [label,index]of [['A0',0],['B1',1],['C0',2],['A0-repeat',0]]){
       const start=logs.length,size=m._malloc(4);let ptr=0;
       try{ptr=m.ccall('wasm_embed_encode_copy','number',['number','string','number'],[contexts[index],text,size]);if(!ptr||m.getValue(size,'i32')!==384)throw new Error('interleaved encode');rows.push({label,vector:Array.from(m.HEAPF32.subarray(ptr/4,ptr/4+384)),logs:logs.slice(start)});}
       finally{if(ptr)m._free(ptr);m._free(size);}
      }
      return rows;
     }finally{for(const ctx of contexts)m._wasm_embed_free(ctx);}
    },cases[0].text);
   }finally{await page.close();}
   if(hash(fs.readFileSync(path.join(root,'old-panel-freeze.json')))!==report.old_panel_freeze_sha256)throw new Error('Old freeze bytes changed');
  }
  report.state='complete';save();if(report.page_errors.length)throw new Error('Browser page errors');
 }catch(error){report.state='failed';report.error=String(error);save();throw error;}
 finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
