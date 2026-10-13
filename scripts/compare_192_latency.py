"""Fresh matched CPU latency comparison of all 192 variants; no energy claims."""
import json
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/latency/resize192';OUT.mkdir(parents=True,exist_ok=True)
MODELS={'baseline':'artifacts/baseline/model_fp32.onnx',
 'resize192':'artifacts/all_data/resize192/model_fp32.onnx',
 'resize192_head_ce':'artifacts/all_data/resize192_head_ce/model_fp32.onnx',
 'resize192_head_ce_w16':'artifacts/all_data/resize192_head_ce_w16/model_weight16.onnx'}
names=list(MODELS)
for rep in range(1,6):
    rotation=(rep-1)%len(names)
    order=names[rotation:]+names[:rotation]
    for name in order:
        path=OUT/f'{name}_rep{rep}.json'
        if path.exists():continue
        print(f'Measuring {name} repetition {rep}',flush=True)
        args=[sys.executable,'-m','edgeprep.benchmark','--model',MODELS[name],'--threads','4',
            '--warmup','100','--runs','1000','--inputs','results/all_data/challenge_data/timing_input.npy',
            '--labels','results/all_data/challenge_data/timing_label.npy','--output',str(path)]
        with (OUT/f'{name}_rep{rep}.stdout.txt').open('w') as stdout,(OUT/f'{name}_rep{rep}.stderr.txt').open('w') as stderr:
            subprocess.run(args,cwd=ROOT,check=True,stdout=stdout,stderr=stderr)
        r=json.loads(path.read_text());print(f'{name}: {r["median_ms"]:.4f} ms',flush=True)
rows=[]
for name in names:
    runs=[json.loads((OUT/f'{name}_rep{i}.json').read_text()) for i in range(1,6)]
    values=[r['median_ms'] for r in runs]
    rows.append({'name':name,'median_ms':statistics.median(values),'run_medians_ms':values,
        'median_min_ms':min(values),'median_max_ms':max(values),'p95_ms':statistics.median(r['p95_ms'] for r in runs),
        'replicates':5,'model_sha256':hashlib.sha256((ROOT/MODELS[name]).read_bytes()).hexdigest(),'threads':4,'warmup':100,'runs_per_rep':1000,
        'provider':'CPUExecutionProvider','spinning_enabled':True,
        'evidence':[f'results/latency/resize192/{name}_rep{i}.json' for i in range(1,6)]})
for r in rows:r['latency_ratio_to_matched_baseline']=r['median_ms']/rows[0]['median_ms']
summary={'protocol':'Serial fresh processes, rotating candidate order, five repetitions; original three retained after high OS variation prompted two additional repetitions, 100 warmups / 1000 timings / four CPU threads, default spinning',
    'scope':'One real timing image; 27000-image accuracy measured separately. No local training during measurements.',
    'limitations':'OS and thermal noise retained; FP16 storage still computes FP32 after ORT constant folding.', 'records':rows}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
