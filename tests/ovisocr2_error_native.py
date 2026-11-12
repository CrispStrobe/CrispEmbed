import ctypes as C,json,sys,hashlib,time
from PIL import Image
lib,model,image,kind=sys.argv[1:]
d=C.CDLL(lib)
def bind(name,args,restype):
 f=getattr(d,name);f.argtypes=args;f.restype=restype;return f
init=bind('qwen35_ocr_init',[C.c_char_p,C.c_int],C.c_void_p)
free=bind('qwen35_ocr_free',[C.c_void_p],None)
recognize=bind('qwen35_ocr_recognize_raw',[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.c_char_p,C.POINTER(C.c_int)],C.c_void_p)
stop=bind('qwen35_ocr_last_stop',[C.c_void_p],C.c_int)
error=bind('qwen35_ocr_last_error',[C.c_void_p],C.c_char_p)
ids=bind('qwen35_ocr_token_ids',[C.c_void_p,C.POINTER(C.c_int)],C.POINTER(C.c_int))
getcap=bind('qwen35_ocr_get_max_tokens',[C.c_void_p],C.c_int)
t=time.monotonic();ctx=init(model.encode(),2);rec={'kind':kind,'init_ok':bool(ctx),'init_s':time.monotonic()-t}
if ctx:
 try:
  im=Image.open(image).convert('RGB');raw=im.tobytes();arr=(C.c_uint8*len(raw)).from_buffer_copy(raw);n=C.c_int();t=time.monotonic()
  ptr=recognize(ctx,arr,*im.size,3,None,C.byref(n));k=C.c_int();tokenptr=ids(ctx,C.byref(k));text=C.string_at(ptr,n.value).decode('utf-8') if ptr else None
  rec.update(default_max_tokens=getcap(ctx),output_pointer_nonnull=bool(ptr),text=text,bytes=n.value,ids=list(tokenptr[:k.value]) if tokenptr else [],stop=stop(ctx),last_error=(error(ctx) or b'').decode(),recognize_s=time.monotonic()-t,rgb_sha256=hashlib.sha256(raw).hexdigest(),width=im.width,height=im.height)
 finally:free(ctx)
print(json.dumps(rec))
