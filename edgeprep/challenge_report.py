"""Consolidate labelled pilot results and plot an honest tradeoff curve."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='results/all_data')
    args=parser.parse_args()
    root=Path(args.root);rows=[]
    baseline=json.loads((root/'baseline_accuracy.json').read_text())
    for p in sorted(root.glob('*_accuracy.json')):
        name=p.name.removesuffix('_accuracy.json')
        acc=json.loads(p.read_text());model=Path(acc['model'])
        recipe=json.loads((model.parent/'recipe.json').read_text())
        row={'candidate':name,'accuracy':acc['accuracy'],'worst_class_accuracy':acc['worst_class_accuracy'],
             'accuracy_retention':acc['accuracy']/baseline['accuracy'],
             'parameters':recipe['parameters'],'macs':recipe['macs'],'size_mib':model.stat().st_size/2**20,
             'examples':acc['examples'],'provisional':acc['provisional'],'model_sha256':acc['model_sha256'],
             'split_sha256':acc['split_sha256'], 'protocol':acc.get('protocol','legacy_split'),
             'evaluation_scope':acc.get('evaluation_scope','split_validation')}
        if acc.get('evaluation_indices',acc.get('validation_indices'))!=baseline.get('evaluation_indices',baseline.get('validation_indices')) or acc.get('protocol')!=baseline.get('protocol'):
            raise ValueError(f'Different evaluation subset for {name}')
        for threads in (1,4):
            bench=root/f'{name}_t{threads}.json'
            if bench.exists():
                value=json.loads(bench.read_text())
                if value['sha256']!=acc['model_sha256']:raise ValueError('Accuracy/timing checkpoint mismatch')
                row[f'latency_t{threads}_ms']=value['median_ms']
                row[f'peak_rss_t{threads}_mib']=value['peak_rss_mib']
                if value.get('energy_j_per_inference') is not None:
                    row[f'estimated_joules_t{threads}']=value['energy_j_per_inference']
        energy=root/f'{name}_energy.json'
        if energy.exists():row['estimated_joules_per_inference']=json.loads(energy.read_text())['energy_j_per_inference']
        rows.append(row)
    baseline_row=next(x for x in rows if x['candidate']=='baseline')
    for row in rows:
        row['parameter_reduction']=baseline_row['parameters']/row['parameters']
        row['onnx_size_reduction']=baseline_row['size_mib']/row['size_mib']
        for threads in (1,4):
            for source,ratio in [(f'latency_t{threads}_ms',f'latency_reduction_t{threads}'),(f'peak_rss_t{threads}_mib',f'ram_reduction_t{threads}'),(f'estimated_joules_t{threads}',f'energy_reduction_t{threads}')]:
                if source in row and source in baseline_row:
                    row[ratio]=baseline_row[source]/row[source]
    (root/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with open(root/'summary.csv','w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    colors=plt.get_cmap('tab20').colors
    handles=[]
    for i,row in enumerate(rows):
        for panel,(ax,x,key) in enumerate([(axes[0],row['size_mib'],'Exported ONNX size (MiB)'),
                         (axes[1],row.get('latency_t4_ms',np.nan),'CPU latency, batch 1 / four threads (ms)')]):
            handle=ax.scatter(x,row['accuracy']*100,s=75,color=colors[i],edgecolors='white',linewidth=.7,label=row['candidate'])
            if panel==0:handles.append(handle)
            ax.set_xlabel(key);ax.set_ylabel('Training-set accuracy (%)' if baseline.get('evaluation_scope')=='training_set' else 'Diagnostic accuracy (%)');ax.grid(alpha=.2)
            ax.set_ylim(max(0,min(r['accuracy'] for r in rows)*100-2),min(100,max(r['accuracy'] for r in rows)*100+1));ax.margins(x=.1)
    scope=baseline.get('evaluation_scope','legacy provisional sample').replace('_',' ')
    fig.suptitle(f"EuroSAT: {len(rows)} variants / {baseline['examples']:,} images / {scope}")
    fig.legend(handles,[r['candidate'] for r in rows],loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.01,'All-data accuracy is training-set performance. Ratios use the supplied student measured on the same laptop.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.22,1,.93));fig.savefig(root/'tradeoff.png',dpi=160);plt.close(fig)
    print('Saved summary.json, summary.csv and tradeoff.png')


if __name__=='__main__':main()
