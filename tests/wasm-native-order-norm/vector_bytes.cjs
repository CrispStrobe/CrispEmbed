'use strict';
// Node-side retention of bytes copied from live WASM memory BEFORE _free.
const crypto=require('crypto');
function retainVector(rawBytes,display){
 if(!Array.isArray(rawBytes)||rawBytes.length!==1536||rawBytes.some(x=>!Number.isInteger(x)||x<0||x>255))throw new Error('Exact1536 raw bytes required');
 if(!Array.isArray(display)||display.length!==384)throw new Error('Display shape differs');
 const bytes=Buffer.from(rawBytes);
 for(let i=0;i<384;++i){const value=bytes.readFloatLE(i*4);if(!Number.isFinite(value)||!Number.isFinite(display[i])||value!==display[i])throw new Error('Raw/display logical values differ');}
 return {raw_base64:bytes.toString('base64'),raw_sha256:crypto.createHash('sha256').update(bytes).digest('hex')};
}
module.exports={retainVector};
