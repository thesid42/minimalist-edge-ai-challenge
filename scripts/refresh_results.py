"""Refresh README, detailed results, sequential log and notebook saved outputs."""
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.next_priority_report import main as report
from edgeprep.experiment_log import main as log
from scripts.saved192_hub_report import main as saved_report
from scripts.fp32_compression_report import main as fp32_report

def read(path):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def main():
    fp32_report();report();saved_report();log()
    baseline=read('results/all_data/baseline_accuracy.json')
    candidate=read('results/all_data/resize192_head_ce_w16_accuracy.json')
    energy=read('results/energy/runtime/summary.json')
    parts=['## Latest completed results\n',f'Refreshed from saved evidence at {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}. [Full results](docs/next_priority_results.md) and [numbered experiment log](experiments/README.md) are regenerated together.\n']
    fp32=read('results/fp32_compression/summary.json')
    if fp32:
        parts+=['**Current direction: structural compression of recovered192, keeping FP32 throughout. Every new candidate uses all 27,000 images for evaluation.** [Full FP32 report](docs/fp32_model_reduction.md).\n',
            '| FP32 model / stage | Parameters | ONNX MiB | Accuracy / 27,000 | Worst class recall |\n|---|---:|---:|---:|---:|']
        for row in fp32['records']:
            acc=f'{row["accuracy_pct"]:.4f}%' if row.get('examples')==27000 else 'pending'
            worst=f'{row["worst_recall_pct"]:.2f}%' if row.get('examples')==27000 else 'pending'
            parts.append(f'| {row["label"]} | {row["parameters"]:,} | {row["size_mib"]:.6f} | {acc} | {worst} |')
        parts.append('\nNew recovery uses GPU FP32 with AMP disabled; export and inference use CPU FP32. The older FP16 comparison below remains historical evidence.\n')
        matched=read('results/fp32_compression/runtime/summary.json')
        if matched:
            parts+=['**New matched FP32 laptop measurements:** four CPU threads, three fresh serial repetitions per model.\n',
                '| FP32 model | Median latency | Range of run medians | Estimated J/image, mean ± SD |\n|---|---:|---:|---:|']
            for r in matched['records']:
                parts.append(f"| {r['name']} | {r['median_ms']:.3f} ms | {r['median_min_ms']:.3f}–{r['median_max_ms']:.3f} ms | {r['energy_j_per_inference']:.6f} ± {r['energy_stdev']:.6f} |")
            best=next(r for r in matched['records'] if r['name']=='head640_head_ce')
            original=next(r for r in matched['records'] if r['name']=='original224_fp32')
            parts.append(f"\nThe 640-channel classifier-recovered model is {(1-best['median_ms']/original['median_ms'])*100:.1f}% faster and uses {(1-best['energy_j_per_inference']/original['energy_j_per_inference'])*100:.1f}% less estimated energy than original224 in this sequence. Accuracy is evaluated separately on all 27,000 images; latency repeats one real input 1,000 times per run. CodeCarbon estimates CPU energy from TDP/load and RAM from a heuristic; GPU is excluded. Small differences versus unpruned192 remain subject to run variability. [Measurements and full interpretation](docs/fp32_model_reduction.md).\n")
        parts.append('**Historical FP16-storage results follow:** these use separate measurements and an earlier winner.\n')

    if baseline and candidate and energy:
        original=next(r for r in energy['records'] if r['name']=='baseline_spin')
        final=next(r for r in energy['records'] if r['name']=='resize192_head_ce_w16_spin')
        bsize=Path(baseline['model']).stat().st_size/2**20
        csize=Path(candidate['model']).stat().st_size/2**20
        parts += ['The measured 192 candidate uses **internal 192 + one full-data classifier recovery epoch + FP16 weight storage / FP32 CPU compute**. External input remains `(1,3,224,224)`.\n',
            '| Matched laptop measurement | Original student | Recovered 192 / weight16 |\n|---|---:|---:|',
            f'| Accuracy on all 27,000 images | {baseline["accuracy"]*100:.4f}% | **{candidate["accuracy"]*100:.4f}%** |',
            f'| Worst class recall | {baseline["worst_class_accuracy"]*100:.2f}% | **{candidate["worst_class_accuracy"]*100:.2f}%** |',
            f'| Portable ONNX size (original FP32 vs candidate FP16 storage) | {bsize:.3f} MiB | **{csize:.3f} MiB** |',
            f'| Four-thread median latency, 3 repetitions | {original["median_ms"]:.3f} ms | **{final["median_ms"]:.3f} ms** |',
            f'| Estimated J/inference, mean of 3 windows | {original["energy_j_per_inference"]:.6f} | **{final["energy_j_per_inference"]:.6f}** |',
            f'\nAccuracy decreases {(baseline["accuracy"]-candidate["accuracy"])*100:.4f} percentage points; the candidate is {(1-final["latency_ratio_to_matched_baseline"])*100:.1f}% faster and uses {(1-final["estimated_energy_ratio_to_matched_baseline"])*100:.1f}% less estimated energy in this matched sequence. CodeCarbon uses CPU TDP/load plus a RAM heuristic on this Windows 10 laptop, with GPU excluded; these are not meter readings. FP16 storage halves file size while ORT folds weights to FP32 for CPU compute; it does not prove a RAM saving. Four threads outperformed 1/2/8 in the measured latency/energy tradeoff. Sleeping workers regressed latency here; the unrecovered 160 model loses too much accuracy (93.00%).\n']
    storage=read('results/storage_comparison/summary.json')
    if storage:
        parts+=['**Controlled weight-storage comparison:** the baseline gets the same FP16 conversion.\n',
            '| Model | FP32 ONNX | FP16-storage ONNX |\n|---|---:|---:|']
        rows={r['name']:r for r in storage['records']}
        for name,label in [('baseline','Unchanged original224'),('recovered192','Recovered192')]:
            parts.append(f'| {label} | {rows[name+"_fp32"]["size_mib"]:.6f} MiB | {rows[name+"_weight16"]["size_mib"]:.6f} MiB |')
        parts.append(f'\nAt identical FP16 storage, recovered192 is **{storage["candidate_minus_baseline_weight16_bytes"]:,} bytes larger**. The roughly 50% portable-file reduction comes from weight precision; the baseline gets it too. Resolution reduction lowers spatial computation and recovery restores accuracy, while parameter count stays unchanged. [Full comparison and accuracy check](docs/matched_weight_storage.md).\n')
    saved=read('best_models/current.json')
    if saved:
        parts.append(f'\n**Saved winner weights:** [{saved["name"]}]({saved["bundle"]}/README.md), with the inference ONNX, full recovered checkpoint, portable recipe, checksums and results. [Current selection](best_models/current.json) / [winner history](best_models/registry.json). Future winners are saved separately; older weights are retained. [Preservation workflow](docs/model_preservation.md).\n')
    direct=read('results/latency/resize192/summary.json')
    if direct:
        parts+=['A separate direct latency comparison of all 192 variants is now complete (five repetitions, four CPU threads, default spinning):\n',
            '| Model | Median latency | Range of run medians |\n|---|---:|---:|']
        for r in direct['records']:
            parts.append(f'| {r["name"]} | {r["median_ms"]:.3f} ms | {r["median_min_ms"]:.3f}–{r["median_max_ms"]:.3f} ms |')
        parts.append('\nThe range shows substantial Windows timing variability. This separate sequence is not the denominator of the energy comparison above. Small FP32/weight16 differences do not establish compute acceleration.\n')
    parts+=['| AI Hub target / execution | Burst latency | Accuracy on the same 1,000 labelled images |\n|---|---:|---:|']
    for folder,label in [('s24_cpu_fp32','S24 family / CPU FP32'),('s24_npu_float_retry','S24 family / NPU float'),('rb3_onnx_npu_w8a8','RB3 Gen 2 Vision Kit / NPU W8A8'),('rb3_onnx_npu_w8a16','RB3 Gen 2 Vision Kit / NPU W8A16')]:
        profile=read(f'results/hub/{folder}/profile.json');score=read(f'results/hub/{folder}/dataset_accuracy.json')
        if profile and score:parts.append(f'| {label} | {profile["execution_summary"]["estimated_inference_time"]/1000:.3f} ms | {score["accuracy"]*100:.1f}% |')
    sample=read('results/hub/resize192_labelled500/manifest.json')
    if sample:
        parts+=['\n**Saved 192 model on Samsung Galaxy S24 — separate 500-image test:**\n',
            '| Execution | Hub profile latency | Accuracy / 500 labelled images |\n|---|---:|---:|']
        for name,label in [('resize192_s24_cpu','CPU / four threads'),('resize192_s24_npu','NPU / burst')]:
            profile=read(f'results/hub/{name}/profile.json');score=read(f'results/hub/{name}/dataset_accuracy.json')
            timing=f'{profile["execution_summary"]["estimated_inference_time"]/1000:.3f} ms' if profile else 'pending'
            acc=f'{score["accuracy"]*100:.2f}% ({score["correct"]}/500)' if score else 'pending'
            parts.append(f'| {label} | {timing} | {acc} |')
        parts.append('\nThe exact saved FP16-storage source is uploaded without extra quantization. Hub compilation expands its target payload to 8.523 MiB; the portable source stays 4.288 MiB. The 500-image sample has 50/class and excludes calibration data; accuracy is diagnostic training-pool accuracy. [Full device report and job evidence](docs/saved192_ai_hub.md).\n')
    parts+=['\nThe first Hub table above uses the original 224 model, not the recovered 192 candidate. Actual NPU placement is recorded per operation. These are training-pool diagnostic accuracies (100/class), disjoint from 512 calibration images; they are not all-data scores or held-out results. W8A16 gains two correct predictions over W8A8 on this sample at higher latency; its worst class is 88% vs W8A8\'s 89%, so it is not a uniform accuracy improvement. Hub power modes expose latency, not measured joules.\n',
        'New training uses `.venv-train-gpu` / RTX 3070 / CUDA; laptop export, evaluation and inference use `.venv` / CPU. AI Hub can use CPU or NPU. A real 100-image, five-step GPU recovery, CPU export and CPU inference verification passed; it is a pipeline check, not a new competition result. **20 functional tests pass; both environments passed dependency checks.**\n',
        'See [GPU setup](docs/gpu_training.md). Refresh the README, log and saved notebook tables after each completed result: `.venv\\Scripts\\python.exe scripts/refresh_results.py`. This reads evidence only and submits no jobs or training.\n']
    path=ROOT/'README.md';text=path.read_text(encoding='utf-8')
    start=text.index('## Latest completed results');end=text.index('## Models and data',start)
    mac_blocks=re.findall(r'<!-- BEGIN (MAC[^>]+) -->.*?<!-- END \1 -->',text[start:end],flags=re.S)
    preserved=[]
    for marker in mac_blocks:
        preserved.append(re.search(r'<!-- BEGIN '+re.escape(marker)+r' -->.*?<!-- END '+re.escape(marker)+r' -->',text[start:end],flags=re.S).group(0))
    path.write_text(text[:start]+'\n'.join(parts)+'\n'+'\n\n'.join(preserved)+'\n'+text[end:],encoding='utf-8')
    print('Refreshed README, result report, numbered folders, registry and notebook')

if __name__=='__main__':main()
