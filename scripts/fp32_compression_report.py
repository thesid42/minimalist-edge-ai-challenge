"""Refresh the FP32-only structural reduction report from saved evidence."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fp32_compression'


def read(path, default=None):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def main():
    builds=read('results/fp32_compression/builds.json')
    if not builds:return
    rows=[]
    for name,label,score_path,recipe_path in [
        ('original224_fp32','Original student / FP32','results/all_data/baseline_accuracy.json','artifacts/baseline/recipe.json'),
        ('parent192_fp32','Recovered192 reference / FP32','results/all_data/resize192_head_ce_accuracy.json','artifacts/all_data/resize192_head_ce/recipe.json')]:
        score=read(score_path);recipe=read(recipe_path)
        rows.append({'name':name,'label':label,'phase':'fp32_structural_comparison','status':'completed','role':'reference',
            'model_sha256':score['model_sha256'],'parameters':recipe['parameters'],'macs':recipe['macs'],'macs_million':recipe['macs']/1e6,
            'size_mib':Path(score['model']).stat().st_size/2**20,'accuracy_pct':score['accuracy']*100,'worst_recall_pct':score['worst_class_accuracy']*100,
            'examples':score['examples'],'scope':'training_set','weight_precision':'FP32','evidence':score_path,
            'implementation':['edgeprep/challenge.py','edgeprep/resolution.py']})
    screen=read('results/fp32_compression/screen_summary.json',builds)
    initial_labels={'late_expansion10':'Late expansion channels / before recovery','head640':'Final 640 channels / before recovery','drop16':'Residual block 16 removed / before recovery'}
    for row in screen['records']:
        row=copy.deepcopy(row)
        row['label']=initial_labels[row['name']];row.update(name='initial_'+row['name'],phase='fp32_structural_comparison',role='initial_pruned',
            label=row['label'],macs_million=row['macs']/1e6)
        rows.append(row)
    recovery_order={'head640_ce_accuracy':0,'drop16_ce_accuracy':1,'head640_head_ce_accuracy':2}
    for path in sorted(OUT.glob('*_ce_accuracy.json'),key=lambda p:recovery_order.get(p.stem,99)):
        score=read(path.relative_to(ROOT));recipe=read(Path(score['model']).parent/'recipe.json')
        if score['examples']!=27000:raise ValueError('All new comparisons require 27000 images')
        rows.append({'name':path.stem.removesuffix('_accuracy'),'label':{'head640_ce':'Final 640 / full-backbone recovery','drop16_ce':'Remove block 16 / full-backbone recovery','head640_head_ce':'Final 640 / classifier-only recovery'}.get(path.stem.removesuffix('_accuracy'),path.stem.removesuffix('_accuracy')),
            'phase':'fp32_structural_comparison','role':'recovered','status':'completed','model_sha256':score['model_sha256'],
            'parameters':recipe['parameters'],'macs':recipe['macs'],'macs_million':recipe['macs']/1e6,
            'size_mib':Path(score['model']).stat().st_size/2**20,'accuracy_pct':score['accuracy']*100,'worst_recall_pct':score['worst_class_accuracy']*100,
            'examples':27000,'scope':'training_set','weight_precision':'FP32','compute_precision':'FP32',
            'training_execution':recipe['recovery']['execution'],'training_examples':recipe['recovery']['available_train_count'],
            'evidence':str(path.relative_to(ROOT)).replace('\\','/'),
            'implementation':['edgeprep/structural.py','edgeprep/challenge.py','scripts/fp32_structural_screen.py']})
    runtime=read('results/fp32_compression/runtime/summary.json',{})
    for row in rows:
        match=next((r for r in runtime.get('records',[]) if r['sha256']==row['model_sha256']),None)
        if match:row.update(median_cpu_4t_ms=match['median_ms'],estimated_cpu_4t_j=match['energy_j_per_inference'],runtime_evidence=match['evidence'])
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'precision_constraint':'FP32 only',
            'evaluation_count_required':27000,'records':rows,
            'protocol':'Train all 27000 on GPU; evaluate all 27000 on CPU. Accuracy is training-set accuracy.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    parts=['# Reduce recovered192 size while keeping FP32\n',
        'The team requested actual parameter/file reduction at identical FP32 precision. The original224 reference is 8.506 MiB and recovered192 reference is 8.511 MiB. All new weights, checkpoints and ONNX computations remain FP32. New recovery uses the RTX 3070 with AMP disabled; laptop inference remains CPU-only. Older FP16 artifacts and evidence remain preserved.\n',
        '**Every new candidate is evaluated on all 27,000 images.** No 500-image model selection is used in this series. This is the organizer all-data training-set protocol, not held-out accuracy. All exports retain `(1,3,224,224)` input and `(1,10)` output.\n',
        '| Model / stage | Parameters | FP32 MiB | Accuracy / 27,000 | Worst class recall |\n|---|---:|---:|---:|---:|']
    for row in rows:
        acc=f'{row["accuracy_pct"]:.4f}%' if row.get('examples')==27000 else 'pending all-data check'
        worst=f'{row["worst_recall_pct"]:.2f}%' if row.get('examples')==27000 else 'pending'
        parts.append(f'| {row["label"]} | {row["parameters"]:,} | {row["size_mib"]:.6f} | {acc} | {worst} |')
    parts+=['\n## What each edit does\n',
        '- **Late expansion pruning:** remove 10% of hidden channels, aligned to eight, only in blocks 14–17. Preserve all early features, bottleneck outputs and residual widths. Rank BN-scaled projection-column norms and physically slice expansion, depthwise and projection tensors.\n- **Final feature reduction:** remove 640 of the final 1,280 feature channels, with dependent BN/classifier inputs sliced consistently. Rank BN-scaled classifier-column norms.\n- **Residual block removal:** replace feature block 16 with Identity. This block has matching input/output width and stride one, so the tensor contract remains valid. Its learned transformations and all their parameters are physically absent.\n',
        'These are structural edits, not zero masks. Exact removed channel indices are saved in recipe.json. A judge recreates the architecture from the supplied student, applies recorded edits, and loads the saved FP32 state. All three initial models passed exact recipe/state replay and PyTorch-to-ONNX parity checks. [Torch-Pruning dependency handling](https://github.com/VainF/Torch-Pruning).\n',
        '## Recovery and measurement\n',
        'Initial accuracy is measured before retraining. Recover the two promising size tradeoffs (head640 and drop16) for one epoch each of full-backbone CE recovery, then test classifier-only recovery from the initial head640 model on all 27,000 images, with BN running statistics frozen. The exported result is evaluated again on all 27,000. Model hashes link scores to files. Isolated, serial four-thread ONNX benchmarks compare original224, unpruned recovered192 and the chosen feature-layer version and recovered block-removal version at FP32; no training runs concurrently. CodeCarbon provides CPU/RAM estimates, not meter readings.\n']
    if runtime.get('records'):
        parts+=['| Matched FP32 runtime | Median ms | Estimated J/inference (mean ± SD) | Repetitions |\n|---|---:|---:|---:|']
        for row in runtime['records']:
            parts.append(f'| {row["name"]} | {row["median_ms"]:.4f} | {row["energy_j_per_inference"]:.6f} ± {row["energy_stdev"]:.6f} | {row["replicates"]} |')
        best=next((r for r in runtime['records'] if r['name']=='head640_head_ce'),None)
        base=next(r for r in runtime['records'] if r['name']=='original224_fp32')
        parent=next(r for r in runtime['records'] if r['name']=='parent192_fp32')
        if best:
            parts.append(f"\nThe classifier-recovered 640-channel model reduces median latency by {(1-best['median_ms']/base['median_ms'])*100:.1f}% and estimated energy per inference by {(1-best['energy_j_per_inference']/base['energy_j_per_inference'])*100:.1f}% versus original224 in this matched sequence. Relative to unpruned recovered192, its latency ratio is {best['median_ms']/parent['median_ms']:.3f} and energy ratio is {best['energy_j_per_inference']/parent['energy_j_per_inference']:.3f}. Small differences within run variability do not establish further acceleration from feature pruning.\n")
        parts.append('Timing uses one real preprocessed image, repeated 1,000 times after 100 warmups per process. The accuracy column comes from the separate complete 27,000-image evaluation. Energy uses three 30-second CPU/RAM CodeCarbon windows, with GPU excluded. Compare energy estimates on this laptop only; these do not measure physical edge-device joules.\n')
    selected=read('best_models/current.json',{})
    if selected.get('name')=='resize192_head640_head_ce_fp32':
        chosen=next(r for r in rows if r['name']=='head640_head_ce')
        base_row=next(r for r in rows if r['name']=='original224_fp32')
        parent_row=next(r for r in rows if r['name']=='parent192_fp32')
        parts+=['\n## Selected FP32 model\n',
            f"**Classifier-only recovery of the 640-channel model is the balanced selection.** Its {chosen['size_mib']:.3f} MiB FP32 ONNX is {(1-chosen['size_mib']/base_row['size_mib'])*100:.1f}% smaller than original224 and {(1-chosen['size_mib']/parent_row['size_mib'])*100:.1f}% smaller than unpruned recovered192, using actual parameter removal. It scores {chosen['accuracy_pct']:.4f}% on all 27,000 training-pool images and {chosen['worst_recall_pct']:.2f}% worst-class recall. It gets four fewer images correct than original224, and eleven more than recovered192. [Saved weights and recipe](../{selected['bundle']}/README.md).\n",
            'Recovery started from the initial head640 pruning result. Train only 6,410 classifier parameters for one GPU FP32 epoch at learning rate 1e-4, batch 32, seed 42, using every image. Backbone tensors and BN running statistics stay identical to the initial pruned model; only classifier weight and bias change. Training took 37.06 seconds. This restores overall accuracy from 98.9963% to 99.1481%.\n',
            'A separate full-backbone epoch at learning rate 5e-5 took 53.54 seconds but reduced head640 accuracy to 98.8815%; keep that negative result. The block-removal recovery took 52.32 seconds and improved 96.4296% to 98.6222%, with a larger 14.3% size reduction and lower worst-class recall of 95.53%. The saved selection favors the better accuracy and class balance. Older winners remain preserved.\n']

    parts+=['\n## Further ways to explore\n',
        '1. **Gradual or data-aware channel pruning:** remove fewer channels per step, recover between steps, or use loss sensitivity over the full training pool to rank removals. Compare gains with the extra optimization time.\n2. **Low-rank factorization of large 1x1 convolutions:** replace a dense matrix with two lower-rank FP32 matrices. This can preserve feature widths while reducing parameters, but adds an operation and needs a measured latency check plus recovery.\n3. **Teacher-assisted recovery:** distill the structurally smaller network from recovered192 or the supplied ResNet50 if hard-label recovery loses too much accuracy. Distillation alone does not shrink a fixed architecture. [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).\n',
        'Dense unstructured zero masks still store the same number of FP32 values. Lower spatial resolution alone changes computation but not weight count. ZIP/download compression is separate from the deployed FP32 model footprint.\n',
        '## Reproduce\n',
        '```powershell\n.venv\\Scripts\\python.exe scripts/fp32_structural_screen.py build\n.venv\\Scripts\\python.exe scripts/fp32_structural_screen.py evaluate\n.venv-train-gpu\\Scripts\\python.exe -m edgeprep.challenge recover --recipe artifacts/fp32_compression/head640/recipe.json --output artifacts/fp32_compression/head640_head_ce --head-only --epochs 1 --train-limit 0 --device cuda --no-amp --workers 2 --batch-size 32 --lr 1e-4\n.venv\\Scripts\\python.exe -m edgeprep.challenge export --recipe artifacts/fp32_compression/head640_head_ce/recipe.json --output artifacts/fp32_compression/head640_head_ce_export\n.venv\\Scripts\\python.exe -m edgeprep.challenge evaluate --model artifacts/fp32_compression/head640_head_ce_export/model_fp32.onnx --limit 0 --threads 4 --output results/fp32_compression/head640_head_ce_accuracy.json\n```\n',
        'For the two full-backbone recovery trials, use the initial head640 or drop16 recipe, omit --head-only, set --lr 5e-5 and choose a separate output folder. Export and evaluate each using the CPU commands above. Exact training indices, state hashes, history and GPU settings are stored in each recipe and the numbered experiment details. Matched runtime is reproduced with scripts/fp32_runtime_compare.py and the two saved candidate paths.\n',
        'Build reconstructs from the public saved FP32 recovered_state.pt in the preserved winner bundle. Obtain the original organizer checkpoint under Models/ as described in the main README. Downloaded data, routine intermediate models and environments remain local. [Summary evidence](../results/fp32_compression/summary.json).\n']
    completed=[r for r in rows if r.get('examples')==27000]
    if completed:
        import os
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        styles={'reference':('#62798a','o'),'initial_pruned':('#d38c35','x'),'recovered':('#19846e','*')}
        for row in completed:
            color,marker=styles[row['role']]
            ax.scatter(row['size_mib'],row['accuracy_pct'],color=color,marker=marker,s=100 if marker=='*' else 55)
            reference=row['role']=='reference'
            ax.annotate(row['name'],(row['size_mib'],row['accuracy_pct']),xytext=(-8 if reference else 5,7 if row['name']!='parent192_fp32' else -15),textcoords='offset points',fontsize=8,ha='right' if reference else 'left')
        ax.set_xlabel('Uncompressed FP32 ONNX file size (MiB)');ax.set_ylabel('Training-set accuracy (%) / all 27000 images')
        ax.set_title('Recovered192: actual structural compression at fixed FP32 precision')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'tradeoff.png',dpi=160);plt.close(fig)
        parts.append('\n![FP32 size and full-data accuracy](../results/fp32_compression/tradeoff.png)\n')
    (ROOT/'docs/fp32_model_reduction.md').write_text('\n'.join(parts),encoding='utf-8')
    print('Updated FP32-only reduction report')


if __name__=='__main__':main()
