"""Summarize new completed evidence; does not run training or cloud jobs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default

def main():
    sections=['\n## Completed measurements\n']
    baseline=read('results/all_data/baseline_accuracy.json')
    sections+=['### All-data accuracy\n','Training-set accuracy on identical 27,000 indices; no held-out generalization claim.\n',
      '| Candidate | Accuracy | Worst class recall | Change vs original |\n|---|---:|---:|---:|']
    for name in ['baseline','resize192','resize160','resize192_head_ce','resize192_head_ce_w16']:
        r=read(f'results/all_data/{name}_accuracy.json')
        if not r:continue
        sections.append(f'| {name} | {r["accuracy"]*100:.4f}% | {r["worst_class_accuracy"]*100:.4f}% | {(r["accuracy"]-baseline["accuracy"])*100:+.4f} pp |')
    storage=read('results/storage_comparison/summary.json')
    if storage:
        sections+=['### Control: identical FP16 storage on the baseline\n',
            '| Model | Storage MiB | All-data accuracy | Worst class recall |\n|---|---:|---:|---:|']
        for r in storage['records']:
            acc=f'{r["accuracy_pct"]:.4f}%' if r['accuracy_pct'] is not None else 'pending'
            worst=f'{r["worst_recall_pct"]:.2f}%' if r['worst_recall_pct'] is not None else 'pending'
            sections.append(f'| {r["name"]} | {r["size_mib"]:.6f} | {acc} | {worst} |')
        sections.append(f'\nThe candidate is {storage["candidate_minus_baseline_weight16_bytes"]:,} bytes larger than the baseline under identical FP16 storage. Neither resolution reduction nor classifier recovery reduces the number of weights. The original FP32-versus-candidate FP16 size ratio is a precision conversion benefit available to both models. [Controlled experiment](matched_weight_storage.md).\n')
    training=read('results/all_data/resize192_head_ce_training.json')
    if training:
        recovery=training['recovery']
        sections.append(f'\nClassifier recovery: one epoch, {recovery["available_train_count"]:,} training images, batch {recovery["batch_size"]}, LR {recovery["lr"]}, {recovery["trained_parameter_count"]:,} trained head parameters. Backbone weights/BN buffers unchanged: {training["backbone_unchanged"]}. Runtime {recovery["seconds"]:.1f} seconds. No teacher or augmentation.\n')
    energy=read('results/energy/runtime/summary.json')
    if energy:
        sections+=['### Latest matched model/runtime measurements\n','Three fresh serial repetitions, four threads, 100 warmups, 1,000 timings and 30-second energy windows. Energy is a TDP/load CPU + RAM heuristic estimate.\n',
          '| Scenario | Median latency | Estimated J/inference (mean ± sample SD) | Latency ratio | Energy ratio |\n|---|---:|---:|---:|---:|']
        for r in energy['records']:
            sections.append(f'| {r["name"]} | {r["median_ms"]:.4f} ms | {r["energy_j_per_inference"]:.6f} ± {r["energy_stdev"]:.6f} | {r["latency_ratio_to_matched_baseline"]:.3f} | {r["estimated_energy_ratio_to_matched_baseline"]:.3f} |')
        sections.append('\nReference is unchanged FP32 with default spinning, freshly remeasured in the same sequence. Sleeping-worker gains require explicit ORT session settings and may trade latency for CPU work. The ONNX file alone does not force judge thread options. Repetition ranges retain OS/thermal variation.\n')
        import os;os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=energy['records'];labels=[r['name'].replace('_','\n') for r in rows]
        fig,axes=plt.subplots(1,2,figsize=(12,5));colors=['#6e7b8b','#91a9be','#197e6b','#39a690']
        axes[0].bar(labels,[r['median_ms'] for r in rows],color=colors);axes[0].set_ylabel('Median inference latency (ms)')
        axes[1].bar(labels,[r['energy_j_per_inference'] for r in rows],yerr=[r['energy_stdev'] for r in rows],capsize=5,color=colors);axes[1].set_ylabel('Estimated joules per inference (mean ± SD)')
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('Laptop ONNX: fixed 224 input / four CPU threads / three repetitions')
        fig.text(.5,.01,'CodeCarbon TDP/CPU-load + system RAM estimate; GPU excluded. Energy is not a hardware-meter reading.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(ROOT/'results/energy/runtime/tradeoff.png',dpi=160);plt.close(fig)
        sections.append('\n![Matched runtime and estimated energy](../results/energy/runtime/tradeoff.png)\n')
    comparison=read('results/latency/resize192/summary.json')
    if comparison:
        sections+=['### Direct 192 latency comparison\n',
            'Separate fresh rotating-order sequence; four CPU threads, default spinning, five repetitions. Use this sequence for comparisons between FP32 and FP16-storage recovered models; do not mix its baseline with older timings.\n',
            '| Model | Median latency | Range of run medians | P95 | Ratio to matched 224 |\n|---|---:|---:|---:|---:|']
        for r in comparison['records']:
            sections.append(f'| {r["name"]} | {r["median_ms"]:.4f} ms | {r["median_min_ms"]:.4f}–{r["median_max_ms"]:.4f} ms | {r["p95_ms"]:.4f} ms | {r["latency_ratio_to_matched_baseline"]:.3f} |')
        sections.append('\nFP16 weights are cast/folded to FP32 on CPU. Small latency differences do not establish FP16 compute acceleration. All repetitions, including noisy results, are retained.\n')
    gpu=read('results/gpu_training/verification.json')
    if gpu:
        sections.append(f'### GPU training verification\n\nSeparate `.venv-train-gpu` / {gpu["training"]["execution"]["device_name"]} / PyTorch {gpu["training"]["execution"]["torch_version"]}. CUDA AMP full-backbone training: {gpu["training_examples"]} real images, {gpu["optimizer_steps"]} steps, {gpu["training"]["seconds"]:.2f} seconds. CPU export and CPU inference passed. This is a pipeline smoke test; it does not replace the selected all-data result. Future training defaults to CUDA without silent CPU fallback. AI Hub CPU/NPU remains independent of the laptop inference rule. [Setup and recipe](gpu_training.md).\n')
    sections+=['### Labelled device accuracy\n','Original224 runs use 1,000 balanced images (100/class); saved192 runs use a separate 500-image sample (50/class). Both exclude the 512 calibration images and remain diagnostic training-pool accuracy. Compare source hashes and sample counts; do not rank accuracy across different samples.\n',
      '| Run | Device | Mode | Internal size | Images | Hub latency | Labelled accuracy | Worst recall |\n|---|---|---|---:|---:|---:|---:|---:|']
    for folder in sorted((ROOT/'results/hub').glob('*/request.json')):
        name=folder.parent.name
        if name=='labelled1000':continue
        request=read(folder.relative_to(ROOT));profile=read(folder.parent.relative_to(ROOT)/'profile.json')
        acc=read(folder.parent.relative_to(ROOT)/'dataset_accuracy.json')
        inherited=False
        if not acc and request.get('accuracy_evidence'):
            acc=read(request['accuracy_evidence']);inherited=bool(acc)
        if not acc and name not in ['rb3_onnx_npu_w8a16','resize192_s24_cpu','resize192_s24_npu']:continue
        time=f'{profile["execution_summary"]["estimated_inference_time"]/1000:.3f} ms' if profile else 'pending'
        score=f'{acc["accuracy"]*100:.2f}%' if acc else 'pending';worst=f'{acc["worst_class_accuracy"]*100:.1f}%' if acc else 'pending'
        mode=request.get('performance_mode','burst')
        if inherited:score+=' (same artifact; burst accuracy)'
        sections.append(f'| {name} | {request["device"]} | {mode} | {request.get("internal_resolution",224)} | {acc["examples"] if acc else "pending"} | {time} | {score} | {worst} |')
    sections.append('\nRemote power modes report latency/memory, not joules. S24 family routing may use different physical units. RB3 W8A8/W8A16 are a same-target precision comparison. Inspect actual operation placement and retain failed DLC jobs.\n')
    path=ROOT/'docs/next_priority_results.md';text=path.read_text(encoding='utf-8-sig').split('\n## Completed measurements')[0]
    path.write_text(text+'\n'.join(sections),encoding='utf-8')
    print('Updated docs/next_priority_results.md')
if __name__=='__main__':main()
