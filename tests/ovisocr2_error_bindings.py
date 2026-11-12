import importlib.util,sys,json
from pathlib import Path
repo,library,model,hello,real=sys.argv[1:]
spec=importlib.util.spec_from_file_location('actual_binding',Path(repo)/'python/crispembed/_binding.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
owner=module.CrispOcrOrchestrator(stages=[dict(source_type=0,engine=14,model_a=model,min_chars=0,min_confidence=0)],router=False,cleanup=False,n_threads=2,lib_path=library)
text=owner.run(hello)['text'];assert text=='Hello OCR\n\n12345'
try:owner.run(real)
except RuntimeError as ex:assert 'image exceeds native patch bound' in str(ex)
else:raise AssertionError('unsupported page became successful blank')
assert owner.run(hello)['text']==text
missing=dict(source_type=0,engine=14,model_a='missing-owned-ocr-fixture.gguf',min_chars=0,min_confidence=0)
legacy=dict(source_type=0,engine=0,min_chars=0,min_confidence=0)
mixed=module.CrispOcrOrchestrator(stages=[missing,legacy],router=False,cleanup=False,n_threads=2,lib_path=library)
try:mixed.run(hello)
except RuntimeError as ex:assert 'initialization failed' in str(ex)
else:raise AssertionError('unknown empty masked reported failure')
standalone=module.CrispOcrOrchestrator(stages=[legacy],router=False,cleanup=False,n_threads=2,lib_path=library)
assert standalone.run(hello)['text']==''
print(json.dumps({'mixed_unknown_rejection_positive':True,'legacy_blank_compatibility_positive':True,'python_pipeline_status_positive':True,'owned_text_preserved':True,'failure_then_success_positive':True,'native_error':'image exceeds native patch bound'}))
