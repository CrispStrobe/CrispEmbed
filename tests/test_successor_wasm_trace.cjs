'use strict';
// Diagnostic modules are hosted test overlays. Packaged modules are served unchanged.
const fs=require('fs'),path=require('path'),http=require('http');
const {chromium}=require('playwright');
const root=path.resolve(process.argv[2]);
const server=http.createServer((req,res)=>{
  const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const target=path.resolve(root,'.'+name);
  if(target!==root&&!target.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  if(name==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>WASM diagnostic consumer</title>');return;}
  res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'application/octet-stream');
  const stream=fs.createReadStream(target);stream.on('error',()=>res.writeHead(404).end());stream.pipe(res);
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  const report={schema:'crispembed-successor-wasm-trace-browser-v1',modes:{},page_errors:[],chromium:browser.version()};
  try{
    for(const mode of ['packaged','diagnostic-ordinary','diagnostic-dump']){
      const page=await browser.newPage();page.on('pageerror',e=>report.page_errors.push(String(e)));
      await page.goto('http://127.0.0.1:'+server.address().port+'/');
      const directory=mode==='packaged'?'crispembed-embed-wasm':'diagnostic';
      await page.addScriptTag({url:'/'+directory+'/crispembed_embed.js'});
      const result=await page.evaluate(async({directory,mode})=>{
        const logs=[];
        const module=await window.CrispEmbedText({locateFile:name=>'/'+directory+'/'+name,printErr:line=>logs.push(line)});
        if(mode!=='packaged'){
          const set=(key,value)=>{if(module.ccall('wasm_trace_setenv','number',['string','string'],[key,value])!==0)throw new Error('setenv failed');};
          set('CRISPEMBED_DEBUG_TOKENS','1');
          if(mode==='diagnostic-dump'){
            set('CRISPEMBED_DUMP_LAYERS','1');set('CRISPEMBED_DUMP_LAYERS_GGUF','/stages.gguf');set('CRISPEMBED_TRACE_POOL','/pool');
          }
        }
        module.FS.writeFile('/model.gguf',new Uint8Array(await(await fetch('/model.gguf')).arrayBuffer()));
        const ctx=module.ccall('wasm_embed_init','number',['string','number'],['/model.gguf',1]);
        if(!ctx)throw new Error('model init failed');
        const size=module._malloc(4);
        try{
          const ptr=module.ccall('wasm_embed_encode_copy','number',['number','string','number'],[ctx,'A small reproducible embedding consumer.',size]);
          const n=module.getValue(size,'i32');if(!ptr||n!==384)throw new Error('output shape');
          const vector=Array.from(module.HEAPF32.subarray(ptr/4,ptr/4+n));module._free(ptr);
          const files={};
          if(mode==='diagnostic-dump')for(const name of ['stages.gguf','pool.mean.f32','pool.final.f32'])files[name]=Array.from(module.FS.readFile('/'+name));
          let operators=null;
          if(mode==='diagnostic-dump'){try{
            // Read the first actual emb_ln_out token from the freshly written F32 GGUF.
            const data=module.FS.readFile('/stages.gguf');const view=new DataView(data.buffer,data.byteOffset,data.byteLength);let offset=0;
            const u32=()=>{const x=view.getUint32(offset,true);offset+=4;return x;};
            const u64=()=>{const x=Number(view.getBigUint64(offset,true));offset+=8;if(!Number.isSafeInteger(x))throw new Error('GGUF integer');return x;};
            const str=()=>{const n=u64();const x=new TextDecoder().decode(data.subarray(offset,offset+n));offset+=n;return x;};
            if(u32()!==0x46554747||u32()!==3)throw new Error('trace GGUF');const nt=u64(),nk=u64();let alignment=32;
            for(let i=0;i<nk;i++){const key=str(),type=u32();let value;if(type===8)value=str();else if(type===4)value=u32();else throw new Error('Unexpected trace metadata type');if(key==='general.alignment')alignment=value;}
            let input=null;
            for(let i=0;i<nt;i++){const name=str(),rank=u32(),shape=[];for(let d=0;d<rank;d++)shape.push(u64());const type=u32(),relative=u64();if(name==='emb_ln_out'){if(type!==0||shape[0]!==384)throw new Error('actual operator input shape/type');input={relative,shape};}}
            if(!input)throw new Error('actual operator input missing');const base=Math.ceil(offset/alignment)*alignment+input.relative;
            const actual=Array.from({length:384},(_,i)=>view.getFloat32(base+4*i,true));
            const weightInfo=await(await fetch('/operator-weight.json')).json();
            const weight=weightInfo.enabled?new Uint8Array(await(await fetch('/operator-weight.bin')).arrayBuffer()):null;
            const wp=weight?module._malloc(weight.length):0;if(weight)module.HEAPU8.set(weight,wp);
            function quantize(label,values){
              const n=values.length,bytes=n/32*34,x=module._malloc(n*4),arch=module._malloc(bytes),ref=module._malloc(bytes),dot=module._malloc(4);
              try{
                module.HEAPF32.set(values,x/4);
                if(module.ccall('wasm_trace_q8','number',Array(4).fill('number'),[x,n,arch,ref])!==bytes)throw new Error('actual Q8_0 conversion failed');
                const a=Array.from(module.HEAPU8.subarray(arch,arch+bytes)),r=Array.from(module.HEAPU8.subarray(ref,ref+bytes));
                const result={label,input:values,arch_bytes:a,reference_bytes:r,byte_mismatches:a.filter((v,i)=>v!==r[i]).length};
                if(label==='zero'&&(a.some(v=>v!==0)||r.some(v=>v!==0)))throw new Error('Q8 zero control');
                // d=1; ties are recorded separately because native SIMD nearest-even differs from roundf ties-away.
                if(label==='fractional-and-ties'&&(r[3]!==2||r[4]!==254))throw new Error('Q8 reference fractional control');
                if(weight&&n===384){
                  const dots={};for(const [kind,pointer]of [['arch',arch],['reference',ref]]){
                    if(module.ccall('wasm_trace_dot','number',Array(5).fill('number'),[weightInfo.stored_type,n,wp,pointer,dot])!==0)throw new Error('actual weight/Q8 dot mapping');
                    dots[kind]=module.HEAPF32[dot/4];
                  }
                  result.actual_weight_dot=dots;result.actual_weight_dot.abs_difference=Math.abs(dots.arch-dots.reference);
                }
                return result;
              }finally{module._free(x);module._free(arch);module._free(ref);module._free(dot);}
            }
            try{
              const fractional=Array(32).fill(0);fractional.splice(0,7,127,1.6,-1.6,1.49,-1.49,2.5,-2.5);
              operators={actual_backend_wasm_simd:module.ccall('wasm_trace_simd','number',[],[]),weight:weightInfo,controls:[quantize('zero',Array(32).fill(0)),quantize('fractional-and-ties',fractional),quantize('actual-first-token-emb_ln_out',actual)],
                limits:['Actual WASM Q8_0 architecture conversion versus pinned roundf reference on identical inputs. One actual query-weight output-channel dot when actual stored type is IQ4_NL; not a native SIMD or full-network fix proof.']};
            }finally{if(wp)module._free(wp);}
          }catch(error){operators={error:String(error)};}}
          return {vector,logs,files,operators};
        }finally{module._free(size);module._wasm_embed_free(ctx);}
      },{directory,mode});
      for(const [name,bytes]of Object.entries(result.files))fs.writeFileSync(path.join(root,'wasm-'+name),Buffer.from(bytes));
      delete result.files;report.modes[mode]=result;
      fs.writeFileSync(path.join(root,'browser-trace.json'),JSON.stringify(report,null,2)+'\n');
      await page.close();
    }
    if(report.page_errors.length)throw new Error('browser page errors');
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
