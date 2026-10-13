"""Consolidate labelled pilot results and plot an honest tradeoff curve."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    root=Path('results/challenge');rows=[]
    baseline=json.loads((root/'baseline_accuracy.json').read_text())
    for p in sorted(root.glob('*_accuracy.json')):
        name=p.name.removesuffix('_accuracy.json')
        acc=json.loads(p.read_text());model=Path(acc['model'])
        recipe=json.loads((model.parent/'recipe.json').read_text())
        row={'candidate':name,'accuracy':acc['accuracy'],'worst_class_accuracy':acc['worst_class_accuracy'],
             'accuracy_retention':acc['accuracy']/baseline['accuracy'],
             'parameters':recipe['parameters'],'macs':recipe['macs'],'size_mib':model.stat().st_size/2**20,
             'examples':acc['examples'],'provisional':acc['provisional'],'model_sha256':acc['model_sha256'],
             'split_sha256':acc['split_sha256']}
        if acc['validation_indices']!=baseline['validation_indices']:
            raise ValueError(f'Different evaluation subset for {name}')
        for threads in (1,4):
            bench=root/f'{name}_t{threads}.json'
            if bench.exists():
                value=json.loads(bench.read_text())
                if value['sha256']!=acc['model_sha256']:raise ValueError('Accuracy/timing checkpoint mismatch')
                row[f'latency_t{threads}_ms']=value['median_ms']
                row[f'peak_rss_t{threads}_mib']=value['peak_rss_mib']
        energy=root/f'{name}_energy.json'
        if energy.exists():row['estimated_joules_per_inference']=json.loads(energy.read_text())['energy_j_per_inference']
        rows.append(row)
    baseline_row=next(x for x in rows if x['candidate']=='baseline')
    for row in rows:
        row['parameter_reduction']=baseline_row['parameters']/row['parameters']
        row['onnx_size_reduction']=baseline_row['size_mib']/row['size_mib']
    (root/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with open(root/'summary.csv','w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,row in enumerate(rows):
        for ax,x,key in [(axes[0],row['size_mib'],'Exported ONNX size (MiB)'),
                         (axes[1],row.get('latency_t1_ms',np.nan),'CPU latency, batch 1 / one thread (ms)')]:
            ax.scatter(x,row['accuracy']*100,s=55)
            ax.annotate(row['candidate'],(x,row['accuracy']*100),fontsize=7,
                        xytext=(4,4 if i%2 else -12),textcoords='offset points')
            ax.set_xlabel(key);ax.set_ylabel('Diagnostic accuracy (%)');ax.grid(alpha=.2)
    fig.suptitle('EuroSAT pilot: 500 balanced provisional validation images')
    fig.text(.5,.01,'Not official scores. Checkpoint training overlap unknown; no recovery except explicitly labelled candidates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(root/'tradeoff.png',dpi=160);plt.close(fig)
    print('Saved summary.json, summary.csv and tradeoff.png')


if __name__=='__main__':main()
