"""FP16 weight storage with explicit FP32 restores; CPU convolution remains FP32."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
from .challenge_data import digest

def convert(source,target):
    graph=onnx.load(source);casts=[];count=0
    occupied={t.name for t in graph.graph.initializer}|{x for n in graph.graph.node for x in n.output}
    for init in graph.graph.initializer:
        if init.data_type!=TensorProto.FLOAT:continue
        original=init.name;name=original+'__storage_fp16'
        if name in occupied:raise ValueError('Weight storage name collision')
        values=numpy_helper.to_array(init)
        if not np.isfinite(values).all() or np.abs(values).max()>np.finfo(np.float16).max:raise ValueError('Weight cannot be represented safely in FP16')
        count+=values.size
        init.CopyFrom(numpy_helper.from_array(values.astype(np.float16),name=name))
        casts.append(helper.make_node('Cast',[name],[original],to=TensorProto.FLOAT,name='RestoreFP32/'+original))
    nodes=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(casts+nodes)
    onnx.checker.check_model(graph);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);onnx.save(graph,target)
    return {'source_onnx':str(Path(source).resolve()),'source_onnx_sha256':digest(source),'output_sha256':digest(target),
      'stored_float_values':int(count),'cast_nodes':len(casts),'weight_storage_precision':'FP16 initializers',
      'compute_precision':'FP32, restored through constant Cast nodes; not FP16 convolution',
      'note':'ORT graph optimization can fold constant casts and expand weights back to FP32 in RAM. Disk savings do not imply RAM or compute savings.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    metadata=convert(a.source,out/'model_weight16.onnx')
    recipe=json.loads((Path(a.source).parent/'recipe.json').read_text(encoding='utf-8'))
    recipe['weight_storage']=metadata
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8');print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
