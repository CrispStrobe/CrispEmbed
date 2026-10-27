# Test utilities adapted from immutable79e350a0; no historical guards executed.
import csv
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from contract_v2 import read_contract,protocol,digest
from diff_adapter import write_pair,collect as collect_ref

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def run(args,log,cwd=None):
    with log.open('w') as stream:
        subprocess.run([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,check=True,timeout=1200)

def typed_fixtures(binary, directory, logs, tensor_schema):
    """Independent writer/schema creates positive and malformed loader inputs."""
    import numpy as np
    import gguf
    cfg = read_contract()['variants']['1.5M']
    arch = 'modernbert_hash_moe_v10'
    rows = tensor_schema(cfg['config'])
    for mutation, expected in [('valid', 'OK'), ('missing', 'Exact 32'), ('nonfinite', 'Nonfinite weight'),
                               ('wrong_type', 'Only lossless F32'), ('wrong_hidden', 'Unsupported 1.5M'),
                               ('unknown', 'Unknown/duplicate tensor'), ('wrong_shape', 'Tensor shape differs')]:
        path = directory / (mutation + '.gguf')
        writer = gguf.GGUFWriter(str(path), arch=arch)
        for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                    'hash_projections', 'max_position_embeddings', 'num_routed_experts', 'num_shared_experts', 'expert_dim', 'top_k']:
            writer.add_uint32(arch + '.' + key, 129 if key == 'hidden_size' and mutation == 'wrong_hidden' else cfg['config'][key])
        for key, value in [('head_dim',64), ('out_dim',128), ('query_marker_token_id',30000), ('document_marker_token_id',30001), ('mask_token_id',4)]:
            writer.add_uint32(arch + '.' + key, value)
        for key, value in [('rope_theta',160000.), ('norm_eps',1e-5), ('hash_norm_eps',1e-5), ('router_renorm_eps',1e-9)]:
            writer.add_float32(arch + '.' + key, value)
        for key, value in [('hidden_activation','gelu_erf'), ('quant_recipe','f32'), ('source_revision',cfg['original']['revision']),
                           ('tokenizer_sidecar_git_blob_sha1',cfg['original']['files']['tokenizer.json']['git_blob_sha1']),
                           ('tokenizer_sidecar_sha256','0'*64)]:
            writer.add_string(arch + '.' + key,value)
        for index, row in enumerate(rows):
            if mutation == 'missing' and index == 0:
                continue
            shape = list(row['shape'])
            if row['transform'] == 'expert_transpose':
                shape[-2:] = reversed(shape[-2:])
            if mutation == 'wrong_shape' and index == 0:
                shape[-1] -= 1
            values = np.zeros(shape, dtype=np.float32)
            if mutation == 'nonfinite' and index == 0:
                values.flat[0] = np.nan
            if mutation == 'wrong_type' and index == 0:
                values = values.astype(np.float16)
            writer.add_tensor('not.a.tensor' if mutation == 'unknown' and index == 0 else row['target'], values)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        run([binary, '--load', path, expected], logs / ('loader-' + mutation + '.log'))


def stages(path):
    import numpy as np
    result, records = {}, {}
    with (path / 'stages.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = row['file']
            if PurePosixPath(name).name != name or row['stage'] in result:
                raise ValueError('Unsafe/duplicate native stage')
            shape = tuple(map(int,row['shape'].split(',')))
            dtype = '<f4' if row['type'] == 'f32' else '<i8'
            data = np.fromfile(path / name, dtype=dtype)
            if data.nbytes != int(row['bytes']) or data.size != __import__('math').prod(shape):
                raise ValueError('Native shape/bytes differ')
            result[row['stage']] = data.reshape(shape)
            if row['type'] == 'f32':
                records[row['stage']] = {'path':path/name,'shape':shape,'sha256':digest(path/name),
                                          'label':path.name+'/'+name}
    return result, records


def compare(a, b, *, integer=False):
    import numpy as np
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Actual stage shape/finiteness differs')
    if integer:
        exact=bool(np.array_equal(a,b))
        return {'exact':exact,'passed':exact}
    difference=b.astype(np.float64)-a.astype(np.float64)
    delta = float(np.max(np.abs(difference),initial=0))
    left, right = a.astype(np.float64).reshape(-1,a.shape[-1]), b.astype(np.float64).reshape(-1,b.shape[-1])
    n1,n2=np.linalg.norm(left,axis=1),np.linalg.norm(right,axis=1)
    cos=np.divide((left*right).sum(axis=1),n1*n2,out=np.zeros_like(n1),where=(n1*n2)!=0)
    cos[(n1==0)&(n2==0)]=1
    worst=float(cos.min(initial=1))
    result={'shape':list(a.shape),'elements':int(a.size),'max_absolute':delta,'min_row_cosine':worst,'mean_absolute':float(np.abs(difference).mean()),
            'mean_error':float(difference.mean()),'rms':float(np.sqrt((difference*difference).mean())),
            'original_l2':float(np.linalg.norm(a.astype(np.float64).ravel())),
            'native_l2':float(np.linalg.norm(b.astype(np.float64).ravel())),
            'passed':delta<=protocol()['gates']['max_absolute'] and worst>=protocol()['gates']['min_row_cosine']}
    for name,value in [('original',a),('native',b)]:
        result.update({name+'_min':float(value.min()),name+'_max':float(value.max()),
                       name+'_abs_min':float(np.abs(value).min()),name+'_abs_max':float(np.abs(value).max())})
    return result


def actual_ref(binary, original, native, report, log, independent):
    """Numeric rc=1 is collected; only structural/process errors interrupt."""
    env={k:v for k,v in os.environ.items() if k not in ('CRISPEMBED_DIFF_COS_THRESHOLD','CRISPEMBED_DIFF_DUMP_DIR')}
    with log.open('w') as stream:
        completed=subprocess.run([str(binary),str(original),str(native),str(report)],stdout=stream,
            stderr=subprocess.STDOUT,env=env,timeout=1200,check=False)
    return collect_ref(completed.returncode,report,independent)


def comparator_controls(binary, directory):
    """Hosted tiny actual-Ref positive/numeric-negative controls, not model gold."""
    import numpy as np
    records=[]
    original=np.asarray([[1.,0.],[0.,1.],[1.,0.]],dtype='<f4')
    for name,expected_exit in [('same',0),('changed_row',1)]:
        output=directory/name;output.mkdir(parents=True)
        native=original.copy()
        if name=='changed_row':native[-1]=[0.,1.]
        specs=[]
        for label,value in [('original',original),('native',native)]:
            path=output/(label+'.f32');path.write_bytes(value.tobytes())
            specs.append({'path':path,'shape':list(value.shape),'sha256':digest(path),'label':label+'.f32'})
        adapters=write_pair(output/'original.gguf',output/'native.gguf',
                            [{'name':'control','original':specs[0],'native':specs[1]}])
        measured={'control':compare(original,native)}
        result=actual_ref(binary,output/'original.gguf',output/'native.gguf',output/'diff.json',output/'diff.log',measured)
        if (result['exit_code']!=expected_exit or not result['all_metrics_agree'] or
                measured['control']['passed']!=(expected_exit==0)):
            raise ValueError('Actual Ref positive/numeric-negative control failed')
        records.append({'control':name,'expected_exit':expected_exit,'actual':result,
                        'metrics':measured,'adapters':adapters})
    write(directory/'controls.json',records)
    return records


def stage_map():
    mapping={'embedding.lookup':'backbone.embeddings.hash_embeddings.output#0',
             'embedding.projection':'backbone.embeddings.projection.output#0',
             'embedding.norm':'backbone.embeddings.norm.output#0'}
    for i in range(2):
        n=f'layer.{i}.'; b=f'backbone.layers.{i}.'
        for native,original in [('attn.norm','attention.norm'),('attn.q','attention.Q_proj'),
                                ('attn.k','attention.K_proj'),('attn.v','attention.V_proj')]:
            mapping[n+native]=b+original+'.output#0'
        for native,original in [('attn.q_rope',f'actual.sdpa.{i}.input.args.0'),
                                ('attn.k_rope_repeat',f'actual.sdpa.{i}.input.args.1'),
                                ('attn.v_repeat',f'actual.sdpa.{i}.input.args.2'),
                                ('attn.sdpa',f'actual.sdpa.{i}.output')]:
            mapping[n+native]=original
        for native,original in [('attn.output_projection','attention.O_proj'),
                                ('attn.residual','attention'),('ffn.norm','ffn_norm'),('ffn.router','ffn.router')]:
            mapping[n+native]=b+original+'.output#0'
        for native,source_suffix in [('router_probabilities','probabilities'),('topk_indices','topk_indices'),
                                     ('topk_weights','renormalized_weights'),('selection_margin','selection_margin')]:
            mapping[n+'ffn.'+native]=f'derived.layers.{i}.ffn.router.{source_suffix}#0'
        for native,original in [('ffn.shared_gate','ffn.shared_gate'),('ffn.shared_up','ffn.shared_up'),
                                ('ffn.shared_down','ffn.shared_down'),('ffn.output','ffn')]:
            mapping[n+native]=b+original+'.output#0'
        mapping[n+'output']=f'backbone.layers.{i}.output#0'
    mapping.update(final_norm='backbone.norm.output#0',dense='pipeline.1.output#0.token_embeddings',
                   normalized='pipeline.3.output#0.token_embeddings')
    return mapping

