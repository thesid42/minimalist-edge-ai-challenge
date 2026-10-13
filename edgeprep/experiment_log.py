"""Build a readable experiment index and notebook from saved evidence; no jobs are run."""
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import zipfile
from pathlib import Path
import pandas as pd

LABELS = {'baseline': 'Unchanged FP32 student', 'qdq_s8': 'Full INT8, signed / MinMax 128', 'qdq_u8': 'Full INT8, unsigned / MinMax 128', 'qdq_s8_256': 'Full INT8 / MinMax 256', 'qdq_percentile': 'Full INT8 / Percentile 256', 'qdq_protected': 'INT8 with first/last/head protected', 'qdq_pointwise': 'Selective INT8 / MinMax 256', 'qdq_percentile512': 'Selective INT8 / Percentile 512', 'prune05': 'Uniform channel pruning 5%', 'prune10': 'Uniform channel pruning 10%', 'expansion05': 'Expansion-only pruning 5%', 'expansion10': 'Expansion-only pruning 10%', 'expansion05_ce': 'Expansion 5% + short CE recovery', 'expansion05_kd': 'Expansion 5% + short teacher KD recovery', 'resize192': 'Internal 192 resize, unchanged weights', 'resize160': 'Internal 160 resize, unchanged weights', 'resize192_head_ce': 'Internal 192 + full-data classifier recovery', 'resize192_head_ce_w16': 'Recovered 192 + FP16 weight storage / FP32 compute'}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"


def read(relative, default=None):
    p = ROOT / relative
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def linked(path):
    return f"[{path}](../../{path})"


def local_records(folder, names, phase):
    rows = []
    for name in names:
        evidence=f"{folder}/{name}_accuracy.json"
        accuracy=read(evidence)
        if not accuracy: continue
        model=Path(accuracy["model"])
        recipe=read(str(model.parent / "recipe.json"), {})
        timing=read(f"results/energy/resolution/{name}_summary.json", {}) if phase=="current_all_data" else {}
        latest_name = {"baseline":"baseline_spin", "resize192_head_ce_w16":"resize192_head_ce_w16_spin"}.get(name)
        if phase=="current_all_data" and latest_name:
            timing=read(f"results/energy/runtime/{latest_name}_summary.json",{}) or timing
        timing=timing or read(f"{folder}/{name}_t4.json", {})
        knobs={k:v for k,v in recipe.items() if k in ["pruning_ratio","round_to","pruning_scope","quantization","recovery","internal_resolution","resize","macs_note","weight_storage"]}
        if knobs.get("recovery"):
            knobs["recovery"]={k:v for k,v in knobs["recovery"].items() if k != "train_indices"}
        rows.append({"name":name,"label":LABELS.get(name,name),"phase":phase,"status":"completed",
                     "accuracy_pct":accuracy["accuracy"]*100,"worst_recall_pct":accuracy["worst_class_accuracy"]*100,
                     "examples":accuracy["examples"],"scope":accuracy.get("evaluation_scope","historical_split_subset"),
                     "parameters":recipe.get("parameters"),"size_mib":model.stat().st_size/2**20 if model.exists() else None,
                     "median_cpu_4t_ms":timing.get("median_ms"),"estimated_cpu_4t_j":timing.get("energy_j_per_inference"),
                     "timing_replicates":timing.get("replicates",1) if timing else None,"timing_evidence":timing.get("evidence",f"{folder}/{name}_t4.json"),"model_sha256":accuracy["model_sha256"],"settings":knobs,"evidence":evidence,
                     "implementation":["edgeprep/challenge.py","edgeprep/export.py","edgeprep/benchmark.py"] + (["edgeprep/resolution.py"] if recipe.get("internal_resolution") else []) + (["edgeprep/weight_storage.py"] if recipe.get("weight_storage") else [])})
    return rows


def hub_records():
    rows=[]
    names = ["s24_cpu_fp32","s24_npu_float","s24_npu_float_retry","rb3_npu_w8a8","rb3_onnx_npu_w8a8"]
    discovered = sorted((ROOT/"results/hub").glob("*/request.json"), key=lambda p:p.stat().st_mtime)
    names += [p.parent.name for p in discovered if p.parent.name not in names]
    for name in names:
        folder=f"results/hub/{name}"
        request=read(folder+"/request.json")
        if not request:continue
        jobs=read(folder+"/jobs.json",{})
        profile=read(folder+"/profile.json")
        check=read(folder+"/numerical_check.json")
        labelled=read(folder+"/dataset_accuracy.json")
        if not labelled and request.get("accuracy_evidence"):labelled=read(request["accuracy_evidence"])
        failures=[v for v in jobs.values() if isinstance(v,dict) and v.get("code")=="FAILED"]
        status="failed" if failures else "completed" if profile and (check or labelled) else "pending"
        summary=profile.get("execution_summary",{}) if profile else {}
        layers=profile.get("execution_detail",[]) if profile else []
        asset_sizes = {}
        for asset in (ROOT/folder).glob("target.*"):
            asset_sizes[asset.name] = {"download_bytes":asset.stat().st_size}
            if asset.suffix == ".zip":
                with zipfile.ZipFile(asset) as archive:
                    asset_sizes[asset.name]["uncompressed_payload_bytes"] = sum(i.file_size for i in archive.infolist())
        rows.append({"name":name,"phase":"remote_exploration","status":status,
                     "device":request["device"],"runtime":request["runtime"],
                     "requested_compute":request["compute_unit_requested"],"precision":request["precision"],
                     "hub_estimated_latency_ms":summary.get("estimated_inference_time",0)/1000 if profile else None,
                     "inference_memory_peak_range_bytes":summary.get("inference_memory_peak_range"),
                     "deployment_assets":asset_sizes, "actual_layer_compute_counts":dict(Counter(x.get("compute_unit","unknown") for x in layers)),
                     "top1_agreement_pct":check["top1_agreement_with_source"]*100 if check else None,
                     "numerical_samples":check.get("examples") if check else None,
                     "accuracy_pct":labelled["accuracy"]*100 if labelled else None,"labelled_samples":labelled.get("examples") if labelled else None,"worst_recall_pct":labelled["worst_class_accuracy"]*100 if labelled else None,"labelled_accuracy_scope":labelled.get("scope") if labelled else None,"labelled_top1_agreement_pct":labelled["top1_agreement_with_source"]*100 if labelled else None,"performance_mode":request.get("performance_mode","burst"),"energy_j":None,"accuracy_inherited_from":request.get("accuracy_evidence"),"failure_details":failures,
                     "settings":request,"jobs":jobs,"evidence":folder,
                     "implementation":["edgeprep/hub_profile.py","edgeprep/hub_auth.py","edgeprep/hub_accuracy.py","edgeprep/hub_modes.py"]})
    return rows


def table(rows, remote=False):
    if remote:
        fields=["name","status","device","precision","performance_mode","hub_estimated_latency_ms","accuracy_pct","worst_recall_pct","labelled_samples","labelled_top1_agreement_pct"]
    else:
        fields=["label","accuracy_pct","worst_recall_pct","size_mib","median_cpu_4t_ms","estimated_cpu_4t_j","examples"]
    return pd.DataFrame([{k:r.get(k) for k in fields} for r in rows]).round(4)


def main():
    OUT.mkdir(exist_ok=True)
    previous=read("experiments/runs.json",{})
    external_records=[r for r in previous.get("records",[]) if r["phase"].startswith("mac_")]
    previous_notebook=read("experiments/Experiment_Log.ipynb",{})
    external_cells=[c for c in previous_notebook.get("cells",[]) if any(k.startswith("mac_cpu") for k in c.get("metadata",{}))]
    historical=local_records("results/challenge", ["baseline","qdq_s8","qdq_u8","qdq_s8_256","qdq_percentile","qdq_protected","qdq_pointwise","prune05","prune10","expansion05","expansion10","expansion05_ce","expansion05_kd"], "historical_pilot")
    names=["baseline","qdq_pointwise","qdq_percentile512"]
    discovered=sorted((ROOT/"results/all_data").glob("*_accuracy.json"),key=lambda p:p.stat().st_mtime)
    names += [p.name.removesuffix("_accuracy.json") for p in discovered if p.name.removesuffix("_accuracy.json") not in names]
    current=local_records("results/all_data",names,"current_all_data")
    remote=hub_records()
    saved_remote=[r for r in remote if r["settings"].get("experiment_group")=="saved192_500"]
    original_remote=[r for r in remote if r not in saved_remote]
    energy=read("results/energy/thread_sweep/summary.json",{})
    energy_rows=[dict(r, name=f'cpu_threads_{r["threads"]}',label=f'FP32 CPU {r["threads"]} threads',phase="runtime_energy",status="completed",evidence="results/energy/thread_sweep/summary.json",implementation=["scripts/thread_energy_sweep.py","edgeprep/benchmark.py"]) for r in energy.get("records",[])]
    runtime=read("results/energy/runtime/summary.json",{})
    runtime_rows=[dict(r, phase="runtime_energy",status="completed",label=r["name"],evidence="results/energy/runtime/summary.json",implementation=["scripts/runtime_energy_sweep.py","edgeprep/benchmark.py"]) for r in runtime.get("records",[])]
    gpu=read("results/gpu_training/verification.json",{})
    gpu_rows=[dict(gpu,name="gpu_training_pipeline",label="RTX 3070 training / CPU inference verification",phase="training_environment",status=gpu.get("status"),evidence="results/gpu_training/verification.json",implementation=["scripts/setup_gpu_training.ps1","scripts/verify_gpu_training.py","edgeprep/challenge.py"])] if gpu else []
    comparison=read("results/latency/resize192/summary.json",{})
    latency_rows=[dict(r,name="latency_"+r["name"],label=r["name"],phase="matched_192_latency",status="completed",implementation=["scripts/compare_192_latency.py","edgeprep/benchmark.py"]) for r in comparison.get("records",[])]
    storage=read("results/storage_comparison/summary.json",{})
    storage_rows=storage.get("records",[])
    fp32=read("results/fp32_compression/summary.json",{})
    fp32_rows=fp32.get("records",[])
    stages=[
      ("01_original_models", "Original models and dataset", "Verified supplied MobileNetV2 and optional ResNet50, exact safe checkpoint loading, fixed 224 input, all 27,000 EuroSAT files and preprocessing. MobileNetV2 has 2,236,682 parameters; ResNet50 is the teacher, never the denominator.", ["results/checkpoint_audit.json","results/dataset_audit.json","edgeprep/challenge_data.py"], []),
      ("02_historical_pilots", "Historical compression and short recovery", "Before the organizer clarification, 13 variants were screened on 500 images from a provisional split. The table follows the experimental progression, not reconstructed timestamps. Full quantization, selective quantization, physical pruning, and short CE/KD recovery are different operations. CE/KD used only 500 training images for one epoch. These rows cannot be mixed with current all-data ratios.",["docs/historical_pilot_analysis.md","results/challenge/summary.json"],historical),
      ("03_measured_baseline", "Our current unchanged-student baseline", "After the organizer clarified all-data training and our own laptop reference, evaluated the unchanged student on all 27,000 images. This is training-set accuracy. Exported FP32 ONNX and measured serial CPU runtime/RAM/estimated energy. No recovery or retraining. The table can display newer matched three-repetition timings; baseline_metrics.json preserves the original 20-second reference measurement.",["baseline_metrics.json","scripts/measure_all_data.ps1"],current[:1]),
      ("04_selective_int8", "Selective INT8 with 256 MinMax samples", "Quantized pointwise convolutions while preserving first/last layers, classifier and depthwise convolutions in FP32. Same learned parameters; smaller weight storage. Calibrated 256 balanced samples and evaluated all 27,000. Smaller does not mean faster: this graph regressed on our laptop CPU.",["ANALYSIS.md","edgeprep/export.py"],[r for r in current if r["name"]=="qdq_pointwise"]),
      ("05_percentile512", "Selective INT8 with 512 Percentile samples", "Same selective precision pattern, now Percentile calibration and 512 balanced samples. This changes two knobs together; it screens a configuration but does not isolate the benefit of either algorithm or sample count. Full-data evaluation and fresh runtime measurements recorded when present.",["results/all_data/calibration512/manifest.json","edgeprep/challenge.py"],[r for r in current if r["name"]=="qdq_percentile512"]),
      ("06_ai_hub", "Real-device CPU/NPU deployment", "Authenticated using the ignored .env key. Retrieved 78 catalogue entries. S24 CPU and NPU exploration, plus RB3 Hub-native W8A8 compilation. First S24 NPU inference failed because compute_unit and an explicit provider were both specified; the retry removes compute_unit from inference options. Remote 32-image agreement is a smoke check. New dataset_accuracy.json files measure labelled accuracy on 1,000 balanced images, separately from all-data laptop accuracy. Preserve failures alongside successful runs.",["docs/ai_hub.md","results/hub/devices.json","edgeprep/hub_profile.py"],original_remote),
      ("07_next_experiments", "Next experiments, not yet measured", "Same-device W8A8 vs W8A16 is now measured in step 06. The equal-precision baseline storage control is now in step 15; use it in future comparisons and measure its matched latency/energy before attributing gains to a particular knob. Prioritize fixed-count MSE vs MinMax calibration, beta Lite-MP, 1-2% expansion pruning with full-data CE/KD, and BN recalibration. Compare every candidate against a reference on the same device/runtime. Full-data classifier-only recovery of the 192 variant is recorded in step 08 when complete; full-backbone pruned-model CE/KD remains pending.",["ANALYSIS.md","docs/ai_hub.md"],[]),
      ("08_internal_resolution", "Lower spatial computation, fixed external input", "PDF section 3 permits resolution reduction while keeping input (1,3,224,224). Resize inside ONNX to 192 or 160, bilinear align_corners=False without antialias, then run unchanged supplied weights. Initial variants have no training or parameter removal. A separate 192 classifier-only CE recovery trains 12,810 head parameters for one epoch on all 27,000 images, keeping backbone and BN statistics frozen. Evaluate all 27,000 images; measure exported ONNX and matched energy. THOP omits bilinear resize overhead. An additional candidate stores FP16 initializers with explicit Cast-to-FP32 nodes: half-sized weight storage, FP32 CPU compute, no inferred RAM saving.",["edgeprep/resolution.py","docs/next_priority_results.md"],[r for r in current if r["name"].startswith("resize")]),
      ("09_energy_measurement", "Repeated CPU latency and estimated energy", "CodeCarbon is the PDF's requested tool. This Windows 10 AMD laptop uses TDP/load estimation. Serial fresh processes, identical weights, 1/2/4/8 threads, three 30-second repetitions. Raw CPU/RAM power and energy CSVs are retained; GPU excluded. Timing uses one real image, dataset accuracy is separate.",["docs/energy_measurement.md","results/energy/thread_sweep/summary.json"],energy_rows),
      ("10_runtime_power", "Recovered model and worker-thread power tradeoff", "Compare default versus sleeping ORT workers on unchanged FP32 and recovered-192 graphs. Three fresh serial repetitions, four threads, identical timing inputs and 30-second CodeCarbon windows. These are estimates. Worker options belong to the runtime and are not enforced by a submitted ONNX file. Accuracy evidence is separate full-data model evaluation.",["scripts/runtime_energy_sweep.py","docs/next_priority_results.md","docs/energy_measurement.md"],runtime_rows),
      ("11_gpu_training", "GPU training environment and CPU inference verification", "New laptop training uses RTX 3070 CUDA, while laptop evaluation/inference remains CPU-only. This does not constrain AI Hub CPU/NPU. Separate matching-version environments, fail-fast CUDA requirement, AMP and CPU-portable checkpoints. Real 100-image full-backbone recovery / five GPU optimizer steps, followed by CPU export and CPU inference, verifies the pipeline only. The earlier 27,000-image classifier recovery occurred on CPU before this preference and remains the selected candidate.",["docs/gpu_training.md","requirements-train-gpu.lock.txt","results/gpu_training/smoke_accuracy.json"],gpu_rows),
      ("12_192_latency", "Direct comparison of 192 model latencies", "Fresh rotating-order CPU timing compares unchanged 224, unchanged internal192, recovered192 FP32 and recovered192 FP16 storage. Five serial processes per model, 100 warmups / 1000 timings / four threads, default spinning. No training in parallel. Timing input is one real image; accuracy evidence remains the separate 27,000-image evaluations. FP16 storage still computes FP32; do not attribute noisy small timing differences to half-precision execution.",["scripts/compare_192_latency.py","results/latency/resize192/summary.json"],latency_rows),
      ("13_mac_cpu", "Matched Mac CPU measurements", "Collaborator measurements on Apple M4 Pro, including full-data accuracy and separate matched CPU latency/estimated energy. Original Mac reports, raw evidence and notebook cells are preserved; do not mix their baselines with Windows or AI Hub.",["results/mac/README.md","experiments/13_mac_cpu/README.md"],[]),
      ("14_saved192_edge_500", "Saved 192 FP16-storage model on an edge phone", "Test the exact saved winner on Samsung Galaxy S24 (Snapdragon 8 Gen 3), separately on CPU and QNN NPU, without additional quantization. Compile from the saved FP16-storage source; inspect compiler output size and actual operation placement. Accuracy uses 500 balanced real images (50/class), disjoint from 512 calibration images, with a local reference from this exact model. Separate microbenchmark profiles give steady inference latency, excluding upload/preprocessing. This is a training-pool diagnostic sample, not 27000-image or held-out accuracy. Jobs can route to separate units of the same named phone.",["docs/saved192_ai_hub.md","results/hub/resize192_labelled500/manifest.json","results/hub/resize192_device.json"],saved_remote),
      ("15_matched_weight_storage", "Baseline and recovered192 at identical weight precision", "Apply the same FP16 initializer storage / FP32 restore conversion to the unchanged baseline. Compare original224 and recovered192 at both FP32 and FP16 storage, verify every rounded baseline initializer and unchanged baseline operations, then measure baseline FP16 accuracy on all 27000 images. A paired 500-image output check is separate. The baseline also becomes about 4.28 MiB, so the selected model has no storage advantage at equal precision; resolution and recovery remove no parameters. Source-versus-compiled deployment size remains separate.",["docs/matched_weight_storage.md","results/storage_comparison/summary.json","results/storage_comparison/baseline_weight16_accuracy.json"],storage_rows),
      ("16_fp32_structural_reduction", "Smaller recovered192 architectures with FP32 weights", "The team now requires FP32 throughout and all 27000 images for every new comparison. Physically reduce late expansion channels, final feature channels or one residual block. Save exact edits and FP32 state, verify replay and export, measure initial full-data accuracy, recover head640 and drop16 with a full-backbone GPU epoch, then recover the initial head640 with only 6410 classifier parameters trained. GPU AMP remains disabled. Reevaluate all three recovered models on the entire pool on CPU. Matched runtime/estimated energy compares original224, parent192, classifier-recovered head640 and recovered drop16 at FP32, with three fresh serial repetitions each. Older FP16 winners remain preserved.",["docs/fp32_model_reduction.md","results/fp32_compression/summary.json","scripts/fp32_structural_screen.py","edgeprep/structural.py"],fp32_rows)
    ]
    for r in gpu_rows:
        r.update(training_device=r["training"]["execution"]["device_name"],amp=r["training"]["execution"]["amp"],seconds=r["training"]["seconds"],inference_provider="CPUExecutionProvider")
    registry={"updated_utc":datetime.now(timezone.utc).isoformat(),"ordering":"Numbered phases and logical experimental progression; original wall-clock timestamps are not reconstructed", "records":historical+current+remote+energy_rows+runtime_rows+gpu_rows+latency_rows+storage_rows+fp32_rows+external_records}
    (OUT/"runs.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    introduction="""# Experiment log - start here

Read the numbered folders in order. Each README explains **what changed, implementation, results and interpretation**, with links to the original evidence. `Experiment_Log.ipynb` presents the same tables with saved outputs; no rerunning training is required to read it. `runs.json` is the machine-readable registry.

Quantization reduces numerical precision and weight storage. Channel pruning physically removes parts of the network. Recovery trains a compressed model to restore accuracy. Device compilation prepares it for a particular runtime.

Two accuracy protocols exist: the historical 500-image pilot and the current 27,000-image training-set evaluation. They are separate tables. Remote runs retain the historical 32-image numerical check and now add a labelled 1,000-image balanced diagnostic subset. That subset excludes 512 calibration images but is not held-out generalization or full-dataset accuracy. All model changes still start from the supplied student.

"""
    saved=read("best_models/current.json")
    if saved:
        introduction+=f'**Current saved winner:** [{saved["name"]}](../{saved["bundle"]}/README.md). The versioned folder contains inference ONNX, recovered training weights, recipe, checksums and metrics. Older winners stay preserved. [Saving future winners](../docs/model_preservation.md).\n\n'
    index=introduction+"| Step | Experiment | Read |\n|---|---|---|\n"
    notebook_cells=[]
    def markdown(text):
        notebook_cells.append({"cell_type":"markdown","id":f"cell-{len(notebook_cells)}","metadata":{},"source":text})
    def code(source, html):
        notebook_cells.append({"cell_type":"code","id":f"cell-{len(notebook_cells)}","metadata":{},"source":source,"execution_count":len([c for c in notebook_cells if c["cell_type"]=="code"])+1,"outputs":[{"output_type":"display_data","data":{"text/html":html},"metadata":{}}]})
    markdown(introduction.replace("# Experiment log - start here", "# Sequential experiment notebook"))
    code('from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display\nproject = Path.cwd()\nif not (project / "edgeprep").exists(): project = project.parent\nregistry = json.loads((project / "experiments/runs.json").read_text())\ndisplay(pd.DataFrame([{k:r.get(k) for k in ["name","phase","status"]} for r in registry["records"]]))',pd.DataFrame([{k:r.get(k) for k in ["name","phase","status"]} for r in registry["records"]]).to_html(index=False))
    for n,(slug,title,explanation,evidence,rows) in enumerate(stages,1):
        folder=OUT/slug;folder.mkdir(exist_ok=True)
        index+=f"| {n:02} | {title} | [Details]({slug}/README.md) |\n"
        body=f"# {n:02}. {title}\n\n{explanation}\n\n"
        markdown(body)
        if slug=="13_mac_cpu":
            notebook_cells.extend(external_cells)
            continue
        if rows:
            df=table(rows,n in (6,14)) if n not in (9,10) else pd.DataFrame([{k:r.get(k) for k in ["threads","replicates","median_latency_ms","mean_estimated_j_per_inference","energy_stdev","latency_ratio_to_matched_4t","estimated_energy_ratio_to_matched_4t"]} for r in rows]).round(6)
            if n==10:df=pd.DataFrame([{k:r.get(k) for k in ["name","median_ms","energy_j_per_inference","energy_stdev","latency_ratio_to_matched_baseline","estimated_energy_ratio_to_matched_baseline"]} for r in rows]).round(6)
            if n==11:
                df=pd.DataFrame([{"name":r["name"],"status":r["status"],"training_examples":r["training_examples"],"optimizer_steps":r["optimizer_steps"],"training_device":r["training"]["execution"]["device_name"],"amp":r["training"]["execution"]["amp"],"seconds":r["training"]["seconds"],"inference_provider":"CPUExecutionProvider"} for r in rows])
                for r in rows:r.update(df.iloc[0].to_dict())
            if n==12:
                df=pd.DataFrame([{k:r.get(k) for k in ["label","median_ms","median_min_ms","median_max_ms","p95_ms","replicates","latency_ratio_to_matched_baseline"]} for r in rows]).round(6)
            if n==15:
                df=pd.DataFrame([{k:r.get(k) for k in ["label","file_bytes","size_mib","stored_float_values","accuracy_pct","worst_recall_pct","examples"]} for r in rows]).round(6)
            if n==16:
                df=pd.DataFrame([{k:r.get(k) for k in ["label","parameters","size_mib","accuracy_pct","worst_recall_pct","examples","median_cpu_4t_ms","estimated_cpu_4t_j"]} for r in rows]).round(6)
            headers=list(df.columns)
            rendered="| "+" | ".join(headers)+" |\n|"+"|".join("---" for _ in headers)+"|\n"
            for values in df.itertuples(index=False,name=None):
                rendered+="| "+" | ".join("pending / n/a" if pd.isna(v) else str(v) for v in values)+" |\n"
            body+=rendered+"\n"
            (folder/"details.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            phase=rows[0]["phase"]
            names=[r["name"] for r in rows];fields=list(df.columns)
            source=f'rows = [r for r in registry["records"] if r["phase"] == {phase!r} and r["name"] in {names!r}]\ndisplay(pd.DataFrame([{{k:r.get(k) for k in {fields!r}}} for r in rows]).round(4))'
            code(source,df.to_html(index=False,na_rep="pending / n/a"))
            if n == 6:
                body += "All latency values in this table are Hub estimates. S24 CPU/NPU comparisons use the same source; numerical agreement retains the 32-image smoke check; labelled accuracy columns use the new 1,000-image diagnostic sample. Actual placement, memory ranges, failure reasons and job IDs are in details.json.\n\n"
            if n==14:
                body+="Labelled sample count is 500. Precision 'none' means no extra Hub quantization; the source already has FP16 weight storage with FP32 restores. Read the source/compiled hashes and deployment asset sizes before claiming precision or compression on the device.\n\n"
            body+="## Implementation and exact settings\n\nOpen [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.\n\n"
            for r in rows:
                body+=f"- **{r.get('label',r['name'])}** (`{r['name']}`): {linked(r['evidence'])}; "+", ".join(linked(x) for x in r["implementation"])+".\n"
        body+="\n## Supporting evidence\n\n"+"\n".join("- "+linked(x) for x in evidence)+"\n"
        (folder/"README.md").write_text(body,encoding="utf-8")
    index+="\nCurrent CPU results:\n\n"+table(current).to_html(index=False,na_rep="pending / n/a")+"\n\nTo refresh this index after more experiments (reads results only):\n\n```powershell\n.venv\\Scripts\\python.exe -m edgeprep.experiment_log\n```\n"
    index += "\nOpen the notebook interactively:\n\n```powershell\npowershell -ExecutionPolicy Bypass -File scripts/notebook.ps1\n```\n\nJupyterLab and the project kernel are installed locally. The notebook reads evidence only; it never reruns training or submits Hub jobs. Timing JSON files also contain a one-image smoke-check field called accuracy; use the separate *_accuracy.json dataset measurements in these tables.\n"
    (OUT/"README.md").write_text(index,encoding="utf-8")
    plot=ROOT/"results/energy/runtime/tradeoff.png"
    if plot.exists():
        markdown("## Latest matched runtime and estimated energy\nSame laptop, four CPU threads and three repetitions. Accuracy is separately measured on all 27,000 images. Remote device results stay separate.")
        notebook_cells.append({"cell_type":"code","id":"tradeoff","metadata":{},"source":'from IPython.display import Image\ndisplay(Image(filename=str(project / "results/energy/runtime/tradeoff.png")))',"execution_count":len([c for c in notebook_cells if c["cell_type"]=="code"])+1,"outputs":[{"output_type":"display_data","data":{"image/png":base64.b64encode(plot.read_bytes()).decode()},"metadata":{}}]})
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3 (project .venv)","language":"python","name":"edgeprep"},"language_info":{"name":"python","version":"3.11"}},"cells":notebook_cells}
    (OUT/"Experiment_Log.ipynb").write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    print(f"Updated {OUT}: {len(registry['records'])} run entries and a notebook with saved outputs")


if __name__ == "__main__":main()
