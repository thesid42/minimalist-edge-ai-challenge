"""Report the saved 192 model's 500-image real-device evaluation from evidence."""
from collections import Counter
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default

def main():
    sample=read('results/hub/resize192_labelled500/manifest.json')
    if not sample:return
    parts=['# Saved 192 model: 500-image edge-device test\n',
        'Source: `best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx`, SHA-256 `'+sample['source_sha256']+'`. This is the preserved recovered192 / FP16-storage winner, not the earlier original224 model.\n',
        'Device: **Samsung Galaxy S24**, Snapdragon 8 Gen 3, Android 14, selected from the live catalogue. CPU and NPU jobs use the same named phone and exact source, but can be routed to different physical units. CPU uses four ONNX Runtime threads; NPU uses QNN in burst mode. No additional W8A8/W8A16 quantization or retraining.\n',
        'Accuracy: **500 real images, 50 per class**, deterministic seed 20261013, excluding all 512 earlier calibration images. This remains a training-pool diagnostic sample, not held-out or all-data accuracy. Local reference logits are generated from this exact saved ONNX. Two 250-image remote jobs per runtime are combined only when both succeed.\n',
        'Latency: separate on-device profile, 100 iterations, batch one, using the Hub-reported steady inference estimate. This excludes upload, image preprocessing and model initialization; it is not the elapsed wall time for the 500-image accuracy job.\n',
        '| Runtime | Profile latency | Labelled accuracy | Correct / 500 | Worst class recall | Agreement with saved source | Actual operation placement |\n|---|---:|---:|---:|---:|---:|---|']
    for name in ['resize192_s24_cpu','resize192_s24_npu']:
        profile=read(f'results/hub/{name}/profile.json');score=read(f'results/hub/{name}/dataset_accuracy.json')
        label='CPU / four threads' if name.endswith('cpu') else 'QNN NPU / burst'
        time=f'{profile["execution_summary"]["estimated_inference_time"]/1000:.3f} ms' if profile else 'pending'
        acc=f'{score["accuracy"]*100:.2f}%' if score else 'pending'
        correct=str(score['correct']) if score else 'pending';worst=f'{score["worst_class_accuracy"]*100:.1f}%' if score else 'pending'
        agreement=f'{score["top1_agreement_with_source"]*100:.1f}%' if score else 'pending'
        placement=str(dict(Counter(x.get('compute_unit','unknown') for x in profile.get('execution_detail',[])))) if profile else 'pending'
        parts.append(f'| {label} | {time} | {acc} | {correct} | {worst} | {agreement} | {placement} |')
    cpu=read('results/hub/resize192_s24_cpu/dataset_accuracy.json')
    npu=read('results/hub/resize192_s24_npu/dataset_accuracy.json')
    if cpu and npu:
        parts+=['\n## Class recall on the 500-image sample\n',
            '| Class | CPU | NPU |\n|---|---:|---:|']
        for label in sample['classes']:
            parts.append(f'| {label} | {cpu["per_class_recall"][label]*100:.1f}% | {npu["per_class_recall"][label]*100:.1f}% |')
        parts.append('\nBoth runtimes make the same three errors as the saved model locally: two PermanentCrop images and one Pasture image. Agreement is reported separately from label-based accuracy.\n')
    parts += ['\n## What happened to weight storage?\n',
        'The uploaded portable source is **4,496,556 bytes / 4.288 MiB**. Both compiled ONNX target packages contain **8,937,126 bytes / 8.523 MiB** uncompressed, including external weights. The target graph has FLOAT initializers: the compiler folded the source FP16-to-FP32 casts and expanded its weight storage. The original saved file remains unchanged. ZIP size is transfer compression and is not the model-size denominator.\n',
        'This confirms a distinction between the smaller portable file and the compiler-produced deployment artifact. The source disk saving is real, but it does not survive this specific Hub compile path. NPU execution is verified by per-operation placement; float precision inside the device kernel is not inferred from file size or the source label.\n',
        'A controlled local experiment applies the same FP16 storage conversion to the original baseline: it becomes 4.283507 MiB versus 4.288250 MiB for this candidate. Equal-precision storage shows no candidate size advantage. [Matched weight-storage evidence](matched_weight_storage.md).\n',
        'Older original224 results were measured on the S24 family and a different 1,000-image sample. They are historical context, not a fresh matched baseline for this exact-phone 192 test. Do not calculate model speedup or compare accuracy across those samples.\n',
        'No energy measurement is claimed. [Qualcomm profile metric definitions](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html). Hub process-memory ranges are retained in profile.json; they are not total board/NPU memory.\n',
        '## Evidence and reproduction\n',
        '- [CPU request, jobs, profile and labelled accuracy](../results/hub/resize192_s24_cpu/)\n- [NPU request, jobs, profile and labelled accuracy](../results/hub/resize192_s24_npu/)\n- [500-image sample manifest](../results/hub/resize192_labelled500/manifest.json)\n- [Live device selection](../results/hub/resize192_device.json)\n',
        '```powershell\n.venv\\Scripts\\python.exe -m edgeprep.hub_accuracy prepare --count 500 --shard-size 250 --sample-dir results/hub/resize192_labelled500 --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx\n.venv\\Scripts\\python.exe -m edgeprep.hub_profile --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx --device "Samsung Galaxy S24" --compute cpu --output results/hub/resize192_s24_cpu\n.venv\\Scripts\\python.exe -m edgeprep.hub_profile --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx --device "Samsung Galaxy S24" --compute npu --output results/hub/resize192_s24_npu\n.venv\\Scripts\\python.exe -m edgeprep.hub_accuracy submit --folder results/hub/resize192_s24_cpu --folder results/hub/resize192_s24_npu --sample-dir results/hub/resize192_labelled500\n.venv\\Scripts\\python.exe -m edgeprep.hub_accuracy collect --folder results/hub/resize192_s24_cpu --folder results/hub/resize192_s24_npu\n```\n',
        'For a new run, choose fresh profile output folders to preserve previous job history. Prepare and labelled-submit are resumable; collect retrieves existing jobs without resubmitting them. Downloaded targets and input/output arrays remain local and ignored by Git.\n']
    (ROOT/'docs/saved192_ai_hub.md').write_text('\n'.join(parts),encoding='utf-8')
    print('Updated saved192 Hub report')

if __name__=='__main__':main()
