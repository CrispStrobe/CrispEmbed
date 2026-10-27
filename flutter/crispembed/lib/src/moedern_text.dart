import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

final class _Result extends Struct {
  @Uint32() external int abiVersion;
  @Uint32() external int structBytes;
  @Int32() external int role;
  @Int32() external int nTokens;
  @Int32() external int nSelected;
  @Int32() external int dimension;
  @Int32() external int expansionCount;
  external Pointer<Int32> ids;
  external Pointer<Uint8> suppliedMask;
  external Pointer<Uint8> effectiveMask;
  external Pointer<Uint8> scoringMask;
  external Pointer<Uint8> expansionPositions;
  external Pointer<Float> allVectors;
  external Pointer<Float> selectedVectors;
}
typedef _InitN=Pointer<Void> Function(Pointer<Utf8>,Int32);
typedef _Init=Pointer<Void> Function(Pointer<Utf8>,int);
typedef _FreeN=Void Function(Pointer<Void>);
typedef _Free=void Function(Pointer<Void>);
typedef _EncodeN=Pointer<_Result> Function(Pointer<Void>,Pointer<Utf8>,Int32);
typedef _Encode=Pointer<_Result> Function(Pointer<Void>,Pointer<Utf8>,int);
typedef _ScoreN=Int32 Function(Pointer<Void>,Pointer<Utf8>,Pointer<Utf8>,Pointer<Float>);
typedef _Score=int Function(Pointer<Void>,Pointer<Utf8>,Pointer<Utf8>,Pointer<Float>);
typedef _ErrorN=Pointer<Utf8> Function(Pointer<Void>);
typedef _Error=Pointer<Utf8> Function(Pointer<Void>);

enum MoEdernRole { query, document }
void _scalarString(String text) {
  final units=text.codeUnits;
  for(var i=0;i<units.length;i++) {
    final u=units[i];
    if(u>=0xd800&&u<=0xdbff) {
      if(++i>=units.length||units[i]<0xdc00||units[i]>0xdfff) throw ArgumentError('Unicode scalar text required');
    } else if(u>=0xdc00&&u<=0xdfff) {throw ArgumentError('Unicode scalar text required');}
  }
  if(text.contains('\u0000')||utf8.encode(text).length>65536) throw ArgumentError('NUL-free text within64KiB required');
}
/// Local packed GGUF only; parent wires shared model resolution separately.
/// Every returned list is copied before another native call can expire storage.
class CrispMoEdern {
  late final DynamicLibrary _lib;
  late final _Free _free;
  late final _Encode _encode;
  late final _Score _score;
  late final _Error _error;
  Pointer<Void> _ctx=nullptr;
  CrispMoEdern(String modelPath,{int nThreads=2,String? libPath}) {
    _scalarString(modelPath);
    if(nThreads<0||nThreads>2)throw ArgumentError('MoEdern threads must be0..2');
    _lib=libPath!=null?DynamicLibrary.open(libPath):Platform.isIOS?DynamicLibrary.process():DynamicLibrary.open(Platform.isWindows?'crispembed.dll':Platform.isMacOS?'libcrispembed.dylib':'libcrispembed.so');
    for(final name in ['init','free','encode','score','error']) {
      if(!_lib.providesSymbol('crispembed_moedern_$name'))throw StateError('Native library does not provide MoEdern text support');
    }
    final init=_lib.lookupFunction<_InitN,_Init>('crispembed_moedern_init');
    _free=_lib.lookupFunction<_FreeN,_Free>('crispembed_moedern_free');
    _encode=_lib.lookupFunction<_EncodeN,_Encode>('crispembed_moedern_encode');
    _score=_lib.lookupFunction<_ScoreN,_Score>('crispembed_moedern_score');
    _error=_lib.lookupFunction<_ErrorN,_Error>('crispembed_moedern_error');
    final path=modelPath.toNativeUtf8();
    try{_ctx=init(path,nThreads);}finally{calloc.free(path);}
    if(_ctx==nullptr)throw StateError(_message());
  }
  String _message(){final p=_error(_ctx);return p==nullptr?'MoEdern operation failed':p.toDartString();}
  void _open(){if(_ctx==nullptr)throw StateError('MoEdern context is closed');}
  Map<String,dynamic> encode(String text,MoEdernRole role) {
    _open();_scalarString(text);final input=text.toNativeUtf8();
    try {
      final p=_encode(_ctx,input,role.index);if(p==nullptr)throw StateError(_message());final r=p.ref;
      if(r.abiVersion!=1||r.structBytes!=sizeOf<_Result>()||r.role!=role.index||r.dimension!=128||r.nSelected<1||r.nSelected>r.nTokens||r.nTokens>512||r.expansionCount!=(role==MoEdernRole.query?r.nTokens:0))throw StateError('MoEdern result ABI/shape differs');
      if(r.ids==nullptr||r.suppliedMask==nullptr||r.effectiveMask==nullptr||r.scoringMask==nullptr||r.allVectors==nullptr||r.selectedVectors==nullptr||(r.expansionCount>0&&r.expansionPositions==nullptr))throw StateError('MoEdern result pointer is null');
      final all=List<double>.from(r.allVectors.asTypedList(r.nTokens*128));
      final selected=List<double>.from(r.selectedVectors.asTypedList(r.nSelected*128));
      if(!all.every((v)=>v.isFinite)||!selected.every((v)=>v.isFinite))throw StateError('Nonfinite MoEdern vectors');
      return {'role':role.name,'dimension':128,'n_tokens':r.nTokens,'n_selected':r.nSelected,
        'ids':List<int>.from(r.ids.asTypedList(r.nTokens)),
        'supplied_mask':List<int>.from(r.suppliedMask.asTypedList(r.nTokens)),
        'effective_mask':List<int>.from(r.effectiveMask.asTypedList(r.nTokens)),
        'scoring_mask':List<int>.from(r.scoringMask.asTypedList(r.nTokens)),
        'expansion_positions':r.expansionCount==0?<int>[]:List<int>.from(r.expansionPositions.asTypedList(r.expansionCount)),
        'all_vectors':all,'selected_vectors':selected};
    }finally{calloc.free(input);}
  }
  double score(String query,String document) {
    _open();_scalarString(query);_scalarString(document);
    final q=query.toNativeUtf8(),d=document.toNativeUtf8(),out=calloc<Float>();
    try {
      if(_score(_ctx,q,d,out)!=1)throw StateError(_message());
      if(!out.value.isFinite)throw StateError('Nonfinite MoEdern score');return out.value;
    }finally{calloc.free(q);calloc.free(d);calloc.free(out);}
  }
  void dispose(){if(_ctx!=nullptr){_free(_ctx);_ctx=nullptr;}}
}
