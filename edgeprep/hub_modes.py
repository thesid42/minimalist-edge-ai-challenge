"""Resumable profiles for power modes of an existing compiled AI Hub artifact."""
import argparse
import json
from pathlib import Path
import qai_hub as hub
from .hub_auth import get_client

def run(folders,collect=False):
    c=get_client()
    for original in folders:
        base=Path(original);request=json.loads((base/'request.json').read_text());jobs=json.loads((base/'jobs.json').read_text())
        for mode in ['balanced','power_saver']:
            out=base.parent/(base.name+'_'+mode);out.mkdir(parents=True,exist_ok=True)
            statepath=out/'jobs.json'
            if not collect:
                if statepath.exists():continue
                compiled=c.get_job(jobs['compile']).get_target_model()
                if compiled is None:raise RuntimeError('No compiled model')
                req=dict(request);req['inference_options']=request['inference_options'].replace('performance_mode=burst','performance_mode='+mode)
                req.update(performance_mode=mode,compiled_job=jobs['compile'],accuracy_evidence=str(base/'dataset_accuracy.json'),validation_note='Profile only; labelled accuracy evidence uses identical compiled artifact under burst mode')
                (out/'request.json').write_text(json.dumps(req,indent=2),encoding='utf-8')
                job=c.submit_inference_job(model=compiled,device=hub.Device(request['device']),profile=True,inputs=None,options=req['inference_options'])
                state={'compile':jobs['compile'],'profile_inference':job.job_id}
            else:
                if not statepath.exists():continue
                state=json.loads(statepath.read_text());job=c.get_job(state['profile_inference']);status=job.get_status()
                state[job.job_id+'_status']={'code':status.code,'message':status.message}
                if status.success and not (out/'profile.json').exists():job.download_profile(str(out/'profile.json'))
                print(out.name,status.code,flush=True)
            statepath.write_text(json.dumps(state,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',action='append',required=True);p.add_argument('--collect',action='store_true');a=p.parse_args();run(a.folder,a.collect)
if __name__=='__main__':main()
