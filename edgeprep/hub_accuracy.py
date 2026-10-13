"""Labelled, resumable AI Hub evaluation on a balanced diagnostic EuroSAT subset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'results/hub/labelled1000'

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def prepare(count=1000, shard_size=250):
    from .challenge_data import dataset, balanced_ids, digest
    import onnxruntime as ort
    if count < 10 or count % 10 or shard_size < 1:
        raise ValueError('Use a positive multiple of ten for a balanced subset')
    ds = dataset(ROOT/'data')
    calibration = json.loads((ROOT/'results/all_data/calibration512/manifest.json').read_text())
    excluded = set(calibration['calibration_indices'])
    available = [i for i in range(len(ds)) if i not in excluded]
    if count > len(available): raise ValueError('Too many diagnostic samples')
    ids = balanced_ids(available, ds.targets, count, seed=20261013)
    labels = np.array([ds.targets[i] for i in ids], dtype=np.int64)
    SAMPLE.mkdir(parents=True, exist_ok=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    model=ROOT/'artifacts/baseline/model_fp32.onnx'
    session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
    logits=[];shards=[]
    for start in range(0,count,shard_size):
        arrays=np.stack([ds[i][0].numpy() for i in ids[start:start+shard_size]])
        name=f'inputs_{start:04d}.npy';np.save(SAMPLE/name,arrays)
        logits.extend(session.run(None,{'images':a[None]})[0][0] for a in arrays)
        shards.append({'start':start,'count':len(arrays),'file':name,'sha256':digest(SAMPLE/name)})
        print(f'Prepared {start+len(arrays)}/{count}',flush=True)
    np.save(SAMPLE/'reference_logits.npy',np.array(logits));np.save(SAMPLE/'labels.npy',labels)
    manifest={'examples':count,'seed':20261013,'classes':ds.classes,'class_support':np.bincount(labels,minlength=10).tolist(),
       'indices':ids,'indices_sha256':hashlib.sha256(np.array(ids,dtype='<i8').tobytes()).hexdigest(),
       'calibration_disjoint':True,'calibration_manifest':'results/all_data/calibration512/manifest.json',
       'source_sha256':digest(model),'sample_order_sha256':calibration['sample_order_sha256'],
       'preprocessing':calibration['preprocessing'],'shards':shards,
       'scope':'training_set_subset','caveat':'Balanced diagnostic sample from organizer all-data pool; not held-out generalization. Excludes these 512 quantization calibration images.'}
    write(SAMPLE/'manifest.json',manifest)

def submit(folder):
    from .hub_auth import get_client
    import qai_hub as hub
    client=get_client();folder=Path(folder)
    manifest=json.loads((SAMPLE/'manifest.json').read_text())
    request=json.loads((folder/'request.json').read_text());jobs=json.loads((folder/'jobs.json').read_text())
    if request['source_sha256'] != manifest['source_sha256']: raise ValueError('Reference source mismatch')
    compiled=client.get_job(jobs['compile']).get_target_model()
    if compiled is None: raise RuntimeError('No compiled target available')
    statefile=folder/'labelled_jobs.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {'manifest':'results/hub/labelled1000/manifest.json','indices_sha256':manifest['indices_sha256'],'jobs':[]}
    if state['indices_sha256'] != manifest['indices_sha256']:raise ValueError('Subset changed; choose a fresh run folder')
    cachefile=SAMPLE/'uploaded_datasets.json'
    cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
    for shard in manifest['shards']:
        key=str(shard['start'])
        if any(x['start']==shard['start'] for x in state['jobs']):continue
        if key in cache:
            if cache[key]['sha256'] != shard['sha256']:raise ValueError('Dataset cache mismatch')
            data=client.get_dataset(cache[key]['id'])
        else:
            values=np.load(SAMPLE/shard['file'],allow_pickle=False)
            data=client.upload_dataset({'images':[a[None] for a in values]},name=f'EuroSAT labelled {key}-{shard["count"]}')
            cache[key]={'id':data.dataset_id,'sha256':shard['sha256']};write(cachefile,cache)
        job=client.submit_inference_job(model=compiled,device=hub.Device(request['device']),inputs=data,
             profile=False,options=request['inference_options'],name=f'{folder.name} labelled {key}')
        state['jobs'].append({'start':shard['start'],'count':shard['count'],'id':job.job_id,'status':'SUBMITTED'})
        write(statefile,state);print(f'{folder.name} {key}: {job.job_id}',flush=True)

def metrics(pred,truth,classes):
    matrix=np.zeros((10,10),dtype=int);np.add.at(matrix,(truth,pred),1)
    recall=matrix.diagonal()/matrix.sum(1)
    return {'accuracy':float(np.mean(pred==truth)),'correct':int((pred==truth).sum()),'examples':len(truth),
      'worst_class_accuracy':float(recall.min()),'per_class_recall':dict(zip(classes,recall.tolist())),
      'class_support':matrix.sum(1).tolist(),'confusion_matrix':matrix.tolist()}

def collect(folder):
    from .hub_auth import get_client
    client=get_client();folder=Path(folder);statefile=folder/'labelled_jobs.json'
    state=json.loads(statefile.read_text());manifest=json.loads((SAMPLE/'manifest.json').read_text())
    if state['indices_sha256']!=manifest['indices_sha256']:raise ValueError('Subset changed')
    reference=np.load(SAMPLE/'reference_logits.npy');truth=np.load(SAMPLE/'labels.npy')
    outputs=[];complete=True
    for item in sorted(state['jobs'],key=lambda x:x['start']):
        path=folder/f'labelled_{item["start"]:04d}.npy'
        if path.exists():values=np.load(path,allow_pickle=False)
        else:
            job=client.get_job(item['id']);status=job.get_status();item['status']=status.code;item['message']=status.message
            print(f'{folder.name} {item["start"]}: {status.code}',flush=True)
            if not status.success:complete=False;continue
            raw=job.download_output_data()
            if not isinstance(raw,dict) or len(raw)!=1:raise ValueError('Expected one output tensor')
            values=np.concatenate(next(iter(raw.values())))
            if values.shape!=(item['count'],10) or not np.isfinite(values).all():raise ValueError('Invalid remote logits')
            np.save(path,values)
        outputs.append(values)
    write(statefile,state)
    if not complete or len(state['jobs'])!=len(manifest['shards']):return False
    remote=np.concatenate(outputs)
    if remote.shape!=reference.shape:raise ValueError('Incomplete outputs')
    result=metrics(remote.argmax(1),truth,manifest['classes'])
    result.update(scope=manifest['scope'],accuracy_caveat=manifest['caveat'],indices_sha256=manifest['indices_sha256'],
        source_sha256=manifest['source_sha256'],reference=metrics(reference.argmax(1),truth,manifest['classes']),
        top1_agreement_with_source=float(np.mean(remote.argmax(1)==reference.argmax(1))),
        max_absolute_logit_error=float(np.abs(remote-reference).max()),calibration_disjoint=True,
        source_correct_target_wrong=int(np.sum((reference.argmax(1)==truth)&(remote.argmax(1)!=truth))),
        source_wrong_target_correct=int(np.sum((reference.argmax(1)!=truth)&(remote.argmax(1)==truth))),
        device=json.loads((folder/'request.json').read_text())['device'],manifest=state['manifest'])
    write(folder/'dataset_accuracy.json',result)
    print(json.dumps({k:result[k] for k in ['accuracy','correct','examples','worst_class_accuracy','top1_agreement_with_source']},indent=2),flush=True)
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','submit','collect'])
    p.add_argument('--folder',action='append',default=[]);p.add_argument('--count',type=int,default=1000)
    p.add_argument('--shard-size',type=int,default=250);a=p.parse_args()
    if a.command=='prepare':prepare(a.count,a.shard_size)
    else:
        if not a.folder:p.error('--folder is required')
        for folder in a.folder:
            (submit if a.command=='submit' else collect)(folder)
if __name__=='__main__':main()
