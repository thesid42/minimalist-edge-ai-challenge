"""Build the official local reference from unchanged student + full-data accuracy + CPU measurements."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/all_data')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',default='baseline_metrics.json')
    p.add_argument('--energy-mode',choices=['measured','tdp_estimate'],default='tdp_estimate')
    args=p.parse_args();root=Path(args.root)
    accuracy=json.loads((root/'baseline_accuracy.json').read_text())
    timing=json.loads((root/f'baseline_t{args.threads}.json').read_text())
    manifest=json.loads((root/'challenge_data/manifest.json').read_text())
    recipe=json.loads((Path(accuracy['model']).parent/'recipe.json').read_text())
    if accuracy['protocol']!='organizer_all_data' or accuracy['examples']!=manifest['examples']:
        raise ValueError('Baseline accuracy must cover the complete organizer all-data protocol')
    if accuracy['model_sha256']!=timing['sha256'] or recipe['pruning_ratio']!=0 or recipe.get('recovery'):
        raise ValueError('Baseline must be the unchanged supplied student, with matching accuracy/timing graph')
    if timing['energy_j_per_inference'] is None:
        raise ValueError('Baseline needs an energy measurement using the same CPU protocol')
    result={'accuracy':accuracy['accuracy'],'params_million':recipe['parameters']/1e6,
            'size_mb':timing['file_bytes']/2**20,'macs_giga':recipe['macs']/1e9,
            'latency_ms':timing['mean_ms'],'median_latency_ms':timing['median_ms'],
            'p95_latency_ms':timing['p95_ms'],'peak_ram_mb':timing['peak_rss_mib'],
            'joules_per_inference':timing['energy_j_per_inference'],
            'worst_class_accuracy':accuracy['worst_class_accuracy'],
            'per_class_recall':accuracy['per_class_recall'],
            'source':'Provided MobileNetV2 measured on this laptop; organizer clarification supersedes PDF placeholders',
            'baseline_role':'unchanged supplied student before recovery or compression',
            'protocol':accuracy['protocol'],'evaluation_scope':'training_set','examples':accuracy['examples'],
            'accuracy_caveat':accuracy['accuracy_caveat'],'sample_order_sha256':manifest['sample_order_sha256'],
            'model_sha256':timing['sha256'],'source_checkpoint_sha256':recipe['source_sha256'],
            'model_format':'FP32 ONNX; size_mb uses 2**20 bytes to match brief code',
            'threads':args.threads,'provider':timing['provider'],'codecarbon_mode':args.energy_mode,
            'measurement_machine':timing['platform']+' / '+timing['cpu'],
            'evidence':{'accuracy':str(root/'baseline_accuracy.json'),'runtime_energy':str(root/f'baseline_t{args.threads}.json')}}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
