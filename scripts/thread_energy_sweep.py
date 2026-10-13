"""Serial, fresh-process CPU thread/energy experiment with matched repeated reference."""
import json
from pathlib import Path
import subprocess
import sys
import statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/energy/thread_sweep'
OUT.mkdir(parents=True,exist_ok=True)
for rep in range(1,4):
    for threads in [4,1,8,2]:
        label=f't{threads}_rep{rep}'
        path=OUT/(label+'.json')
        if path.exists():continue
        print(f'Running {label}: 1000 timing iterations + 30 seconds estimated energy',flush=True)
        subprocess.run([sys.executable,'-m','edgeprep.benchmark','--model','artifacts/baseline/model_fp32.onnx',
            '--size','224','--threads',str(threads),'--runs','1000','--warmup','100',
            '--inputs','results/all_data/challenge_data/timing_input.npy',
            '--labels','results/all_data/challenge_data/timing_label.npy','--energy-seconds','30',
            '--energy-output',str(OUT/label),'--output',str(path)],cwd=ROOT,check=True,
            stdout=(OUT/(label+'.stdout.txt')).open('w'),stderr=(OUT/(label+'.stderr.txt')).open('w'))
        r=json.loads(path.read_text());print(f'{label}: {r["median_ms"]:.3f} ms, {r["energy_j_per_inference"]:.6f} estimated J',flush=True)
records=[]
for t in [1,2,4,8]:
    runs=[json.loads((OUT/f't{t}_rep{r}.json').read_text()) for r in range(1,4)]
    energies=[x['energy_j_per_inference'] for x in runs]
    records.append({'threads':t,'replicates':3,'median_latency_ms':statistics.median(x['median_ms'] for x in runs),
      'mean_estimated_j_per_inference':statistics.mean(energies),'energy_stdev':statistics.stdev(energies),
      'energy_min':min(energies),'energy_max':max(energies),'codecarbon_cpu_mode':next((x['codecarbon_cpu_mode'] for x in runs if x['codecarbon_cpu_mode']!='unknown'),'unknown'),
      'cpu_tdp_w':next((x['codecarbon_cpu_tdp_w'] for x in runs if x['codecarbon_cpu_tdp_w'] is not None),None),'model_sha256':runs[0]['sha256']})
ref=next(x for x in records if x['threads']==4)
for x in records:
    x['latency_ratio_to_matched_4t']=x['median_latency_ms']/ref['median_latency_ms']
    x['estimated_energy_ratio_to_matched_4t']=x['mean_estimated_j_per_inference']/ref['mean_estimated_j_per_inference']
result={'scope':'same laptop process CPU TDP/load estimate plus system RAM heuristic, GPU excluded',
 'model_changes':'none; all scenarios use identical FP32 student','accuracy_evidence':'results/all_data/baseline_accuracy.json',
 'timing_sample':'one fixed real image; dataset accuracy separately evaluated on all 27000',
 'protocol':'Fresh process per scenario; serial 4,1,8,2 threads; three repetitions; 100 warmups,1000 timings,30 second inference energy loops',
 'records':records,'reference_threads':4}
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
