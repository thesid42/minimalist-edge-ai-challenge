"""Matched model/runtime experiment: default spinning versus sleeping workers."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/energy/runtime';OUT.mkdir(parents=True,exist_ok=True)
scenarios={'baseline_spin':('artifacts/baseline/model_fp32.onnx',False),
 'baseline_sleep':('artifacts/baseline/model_fp32.onnx',True),
 'resize192_head_ce_w16_spin':('artifacts/all_data/resize192_head_ce_w16/model_weight16.onnx',False),
 'resize192_head_ce_w16_sleep':('artifacts/all_data/resize192_head_ce_w16/model_weight16.onnx',True)}
for rep in range(1,4):
    for name,(model,disable) in scenarios.items():
        label=f'{name}_rep{rep}';path=OUT/(label+'.json')
        if path.exists():continue
        print(f'Measuring {label}',flush=True)
        args=[sys.executable,'-m','edgeprep.benchmark','--model',model,'--threads','4','--runs','1000','--warmup','100',
         '--inputs','results/all_data/challenge_data/timing_input.npy','--labels','results/all_data/challenge_data/timing_label.npy',
         '--energy-seconds','30','--energy-output',str(OUT/label),'--output',str(path)]
        if disable:args+=['--disable-spinning']
        with (OUT/(label+'.stdout.txt')).open('w') as stdout,(OUT/(label+'.stderr.txt')).open('w') as stderr:
            subprocess.run(args,cwd=ROOT,check=True,stdout=stdout,stderr=stderr)
        r=json.loads(path.read_text());print(f'{name}: {r["median_ms"]:.3f} ms, {r["energy_j_per_inference"]:.6f} estimated J',flush=True)
records=[]
for name in scenarios:
    runs=[json.loads((OUT/f'{name}_rep{r}.json').read_text()) for r in range(1,4)]
    r=dict(runs[0]);energies=[x['energy_j_per_inference'] for x in runs]
    r.update(name=name,replicates=3,median_ms=statistics.median(x['median_ms'] for x in runs),
       mean_ms=statistics.mean(x['mean_ms'] for x in runs),p95_ms=statistics.median(x['p95_ms'] for x in runs),
       peak_rss_mib=max(x['peak_rss_mib'] for x in runs),energy_j_per_inference=statistics.mean(energies),
       energy_stdev=statistics.stdev(energies),energy_min=min(energies),energy_max=max(energies),
       energy_estimation_note='Process CPU TDP/load plus system RAM heuristic, GPU excluded. Default Windows Balanced plan, AC power.',
       aggregate_note='Median of run medians/P95, mean of means/energy, max of process RSS peaks',
       evidence=[f'results/energy/runtime/{name}_rep{i}.json' for i in range(1,4)])
    records.append(r)
ref=records[0]
for r in records:
    r['latency_ratio_to_matched_baseline']=r['median_ms']/ref['median_ms']
    r['estimated_energy_ratio_to_matched_baseline']=r['energy_j_per_inference']/ref['energy_j_per_inference']
    (OUT/f'{r["name"]}_summary.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
result={'protocol':'Serial fresh processes, three repetitions; baseline default, baseline sleep, recovered192 default, recovered192 sleep; four threads,100 warmups,1000 timings,30-second energy loops',
 'scope':'Steady inference CPU timing and TDP/load CPU + system RAM estimate; GPU excluded',
 'limitation':'Thread options require runtime configuration; a submitted ONNX alone does not force judge session options. Power/OS noise retained in repetition ranges.',
 'records':records}
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([{k:r[k] for k in ['name','median_ms','energy_j_per_inference','latency_ratio_to_matched_baseline','estimated_energy_ratio_to_matched_baseline']} for r in records],indent=2),flush=True)
