"""Consolidate labelled pilot results and plot an honest tradeoff curve."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    root=Path('results/challenge');rows=[]
    baseline=json.loads((root/'baseline_accuracy.json').read_text())
    reference=json.loads(Path('baseline_metrics.json').read_text()) if Path('baseline_metrics.json').exists() else None
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
        if reference:
            row['pdf_reference_accuracy_retention']=row['accuracy']/reference['accuracy']
            row['pdf_reference_parameter_reduction']=reference['params_million']*1e6/row['parameters']
            row['pdf_reference_size_reduction']=reference['size_mb']/row['size_mib']
            row['pdf_reference_mac_reduction']=reference['macs_giga']*1e9/row['macs']
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
                         (axes[1],row.get('latency_t1_ms',np.nan),'CPU latency, batch 1 / one thread (ms)')]):
            handle=ax.scatter(x,row['accuracy']*100,s=75,color=colors[i],edgecolors='white',linewidth=.7,label=row['candidate'])
            if panel==0:handles.append(handle)
            ax.set_xlabel(key);ax.set_ylabel('Diagnostic accuracy (%)');ax.grid(alpha=.2)
            ax.set_ylim(10,103);ax.margins(x=.1)
    fig.suptitle('EuroSAT pilot: 13 variants / 500 balanced provisional validation images')
    fig.legend(handles,[r['candidate'] for r in rows],loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.01,'Diagnostic only: checkpoint training overlap unknown. Accuracy and efficiency use measured student; PDF-reference ratios are also in CSV.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.22,1,.93));fig.savefig(root/'tradeoff.png',dpi=160);plt.close(fig)
    print('Saved summary.json, summary.csv and tradeoff.png')


if __name__=='__main__':main()
