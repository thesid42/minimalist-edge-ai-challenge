"""Matched serial FP32 CPU latency and estimated energy, with three repetitions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fp32_compression/runtime'


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,action='append');p.add_argument('--name',required=True,action='append');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    scenarios={'original224_fp32':'artifacts/baseline/model_fp32.onnx',
               'parent192_fp32':'artifacts/all_data/resize192_head_ce/model_fp32.onnx'}
    if len(args.name)!=len(args.candidate):raise ValueError('Each candidate needs a unique name')
    scenarios.update(zip(args.name,args.candidate))
    for rep in range(1,4):
        names=list(scenarios);names=names[rep-1:]+names[:rep-1]
        for name in names:
            model=Path(scenarios[name]);path=OUT/f'{name}_rep{rep}.json'
            expected=hashlib.sha256(model.read_bytes()).hexdigest()
            if path.exists():
                if json.loads(path.read_text())['sha256']!=expected:raise ValueError('Existing benchmark belongs to other weights')
                continue
            print(f'Measuring FP32 {name}, repetition {rep}',flush=True)
            command=[sys.executable,'-m','edgeprep.benchmark','--model',str(model),'--threads','4','--runs','1000','--warmup','100',
                '--inputs','results/all_data/challenge_data/timing_input.npy','--labels','results/all_data/challenge_data/timing_label.npy',
                '--energy-seconds','30','--energy-output',str(OUT/f'{name}_rep{rep}'),'--output',str(path)]
            with (OUT/f'{name}_rep{rep}.stdout.txt').open('w') as out,(OUT/f'{name}_rep{rep}.stderr.txt').open('w') as err:
                subprocess.run(command,cwd=ROOT,check=True,stdout=out,stderr=err)
            value=json.loads(path.read_text())
            print(f'{name}: {value["median_ms"]:.4f} ms, {value["energy_j_per_inference"]:.6f} estimated J',flush=True)
    records=[]
    for name in scenarios:
        runs=[json.loads((OUT/f'{name}_rep{i}.json').read_text()) for i in range(1,4)]
        value=dict(runs[0]);energies=[r['energy_j_per_inference'] for r in runs]
        value.update(name=name,replicates=3,median_ms=statistics.median(r['median_ms'] for r in runs),
            median_min_ms=min(r['median_ms'] for r in runs),median_max_ms=max(r['median_ms'] for r in runs),
            p95_ms=statistics.median(r['p95_ms'] for r in runs),energy_j_per_inference=statistics.mean(energies),
            energy_stdev=statistics.stdev(energies),peak_rss_mib=max(r['peak_rss_mib'] for r in runs),
            evidence=[f'results/fp32_compression/runtime/{name}_rep{i}.json' for i in range(1,4)])
        records.append(value)
    for row in records:
        for reference in records[:2]:
            row['latency_ratio_to_'+reference['name']]=row['median_ms']/reference['median_ms']
            row['energy_ratio_to_'+reference['name']]=row['energy_j_per_inference']/reference['energy_j_per_inference']
        (OUT/f'{row["name"]}_summary.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'protocol':'All FP32; serial fresh processes, rotating order, 3 repetitions, 4 threads, 100 warmups, 1000 timings and 30-second CodeCarbon energy windows; no training concurrently',
        'energy_caveat':'CPU load/TDP + RAM heuristic, GPU excluded; no hardware-meter claim','records':records},indent=2),encoding='utf-8')
    print('Matched FP32 runtime comparison complete',flush=True)


if __name__=='__main__':main()
