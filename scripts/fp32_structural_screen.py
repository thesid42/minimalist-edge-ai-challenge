"""Build three smaller FP32 recovered192 architectures and evaluate all 27000 images.

Architecture/export and inference run in CPU .venv. Training is a separate GPU step.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort
import torch
from thop import profile
from edgeprep.challenge import restore_recipe
from edgeprep.challenge_data import dataset, balanced_ids, digest
from edgeprep.export import export_fp32
from edgeprep.structural import apply_edits, plan_edits

OUT=ROOT/'results/fp32_compression'
PARENT=ROOT/'best_models/resize192_head_ce_w16_e391533162d7/recipe.json'
PARENT_FP32=ROOT/'artifacts/all_data/resize192_head_ce/model_fp32.onnx'
MODES=['late_expansion10','head640','drop16']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    if not PARENT_FP32.exists():
        parent, parent_recipe = restore_recipe(PARENT)
        PARENT_FP32.parent.mkdir(parents=True, exist_ok=True)
        export_fp32(parent, PARENT_FP32, fixed_batch=True)
    assert digest(PARENT_FP32) == read(PARENT)['weight_storage']['source_onnx_sha256']
    records=[]
    for mode in MODES:
        model,recipe=restore_recipe(PARENT)
        parent_parameters=sum(p.numel() for p in model.parameters())
        parent_hash=recipe['recovered_sha256']
        edits=plan_edits(model,mode)
        model=apply_edits(model,edits,192)
        folder=ROOT/'artifacts/fp32_compression'/mode
        folder.mkdir(parents=True,exist_ok=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in state.values())
        torch.save(state,folder/'recovered_state.pt')
        recipe['parent_recovery']={k:v for k,v in recipe.pop('recovery',{}).items() if k!='train_indices'}
        recipe.update(structural_edits=edits,recovered_checkpoint=str(folder/'recovered_state.pt'),
            recovered_sha256=digest(folder/'recovered_state.pt'),training='Inherited recovered192 FP32 weights, physically reduced; no post-pruning training yet',
            parameters=sum(p.numel() for p in model.parameters()),quantization=None,
            compression={'parent_recipe':str(PARENT.relative_to(ROOT)), 'parent_checkpoint_sha256':parent_hash,
                         'parent_onnx_sha256':digest(PARENT_FP32),
                         'mode':mode,'weight_precision':'FP32','compute_precision':'FP32','parent_parameters':parent_parameters,
                         'torch_pruning_version':importlib.metadata.version('torch-pruning')})
        recipe.pop('weight_storage',None)
        recipe['macs'],_=profile(copy.deepcopy(model),inputs=(torch.zeros(1,3,224,224),),verbose=False)
        recipe['fp32_parity_max_error']=export_fp32(model,folder/'model_fp32.onnx',fixed_batch=True)
        graph=onnx.load(folder/'model_fp32.onnx')
        assert all(i.data_type in (TensorProto.FLOAT,TensorProto.INT64,TensorProto.INT32) for i in graph.graph.initializer)
        (folder/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
        replay,_=restore_recipe(folder/'recipe.json')
        for key,value in state.items():torch.testing.assert_close(value,replay.state_dict()[key],rtol=0,atol=0)
        with torch.no_grad():
            sample=torch.randn(1,3,224,224)
            torch.testing.assert_close(model(sample),replay(sample),rtol=0,atol=0)
        record={'name':mode,'label':mode,'phase':'fp32_structural_initial','status':'built',
            'model':str(folder/'model_fp32.onnx'),'model_sha256':digest(folder/'model_fp32.onnx'),
            'parameters':recipe['parameters'],'macs':recipe['macs'],'file_bytes':(folder/'model_fp32.onnx').stat().st_size,
            'size_mib':(folder/'model_fp32.onnx').stat().st_size/2**20,
            'parameter_reduction_pct':(1-recipe['parameters']/parent_parameters)*100,
            'size_reduction_vs_parent_pct':(1-(folder/'model_fp32.onnx').stat().st_size/PARENT_FP32.stat().st_size)*100,
            'weight_precision':'FP32','compute_precision':'FP32','recipe_replay_verified':True,
            'fp32_parity_max_error':recipe['fp32_parity_max_error'],'evidence':f'results/fp32_compression/{mode}_accuracy.json',
            'implementation':['scripts/fp32_structural_screen.py','edgeprep/structural.py','edgeprep/challenge.py']}
        records.append(record)
        print(json.dumps({k:record[k] for k in ['name','parameters','size_mib','parameter_reduction_pct','size_reduction_vs_parent_pct'] }),flush=True)
    (OUT/'builds.json').write_text(json.dumps({'records':records,'parent_recipe':str(PARENT),'precision':'FP32'},indent=2),encoding='utf-8')


def evaluate_all():
    torch.set_num_threads(1)
    metadata=read(OUT/'builds.json');records=metadata['records']
    ds=dataset('data');ids=balanced_ids(list(range(len(ds))),ds.targets,0)
    if len(ids)!=27000:raise ValueError('Expected all 27000 EuroSAT images')
    index_hash=hashlib.sha256(np.array(ids,dtype='<i8').tobytes()).hexdigest()
    sessions={}
    for row in records:
        assert digest(row['model'])==row['model_sha256']
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(row['model'],options,providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,3,224,224]
        sessions[row['name']]=session
    matrices={name:np.zeros((10,10),dtype=np.int64) for name in sessions}
    started=time.perf_counter()
    for position,index in enumerate(ids,1):
        x,y=ds[index];input_array=x.numpy()[None]
        for name,session in sessions.items():
            output=session.run(None,{session.get_inputs()[0].name:input_array})[0]
            assert output.shape==(1,10) and np.isfinite(output).all()
            matrices[name][y,int(output.argmax())]+=1
        if position%1000==0 or position==len(ids):
            progress={'examples_per_model':position,'target_per_model':27000,'models':list(sessions),
                      'elapsed_seconds':time.perf_counter()-started,'status':'running' if position<len(ids) else 'completed'}
            (OUT/'screen_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            print(f'All three FP32 candidates: {position}/27000 images each',flush=True)
    for row in records:
        matrix=matrices[row['name']];support=matrix.sum(1);recall=matrix.diagonal()/support
        result={'model':row['model'],'model_sha256':row['model_sha256'],'accuracy':float(matrix.trace()/matrix.sum()),
            'correct':int(matrix.trace()),'examples':len(ids),'worst_class_accuracy':float(recall.min()),
            'per_class_recall':dict(zip(ds.classes,recall.tolist())),'class_support':support.tolist(),'confusion_matrix':matrix.tolist(),
            'evaluation_indices':ids,'evaluation_indices_sha256':index_hash,'split_sha256':None,
            'evaluation_scope':'training_set','protocol':'organizer_all_data','provisional':False,
            'full_evaluation_count':27000,'accuracy_caveat':'All-data training-set accuracy; no held-out estimate',
            'provider':'CPUExecutionProvider','threads':1,'preprocessing_threads':1,
            'evaluation_seconds_all_three':time.perf_counter()-started}
        (OUT/(row['name']+'_accuracy.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        row.update(status='completed',accuracy_pct=result['accuracy']*100,worst_recall_pct=result['worst_class_accuracy']*100,
                   examples=27000,scope='training_set')
        print(json.dumps({k:row[k] for k in ['name','size_mib','accuracy_pct','worst_recall_pct','examples']}),flush=True)
    metadata.update(records=records,updated_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'screen_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','evaluate']);args=p.parse_args()
    build() if args.command=='build' else evaluate_all()


if __name__=='__main__':main()
