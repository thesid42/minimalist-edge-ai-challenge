"""Build a readable experiment index and notebook from saved evidence; no jobs are run."""
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import zipfile
from pathlib import Path
import pandas as pd

LABELS = {'baseline': 'Unchanged FP32 student', 'qdq_s8': 'Full INT8, signed / MinMax 128', 'qdq_u8': 'Full INT8, unsigned / MinMax 128', 'qdq_s8_256': 'Full INT8 / MinMax 256', 'qdq_percentile': 'Full INT8 / Percentile 256', 'qdq_protected': 'INT8 with first/last/head protected', 'qdq_pointwise': 'Selective INT8 / MinMax 256', 'qdq_percentile512': 'Selective INT8 / Percentile 512', 'prune05': 'Uniform channel pruning 5%', 'prune10': 'Uniform channel pruning 10%', 'expansion05': 'Expansion-only pruning 5%', 'expansion10': 'Expansion-only pruning 10%', 'expansion05_ce': 'Expansion 5% + short CE recovery', 'expansion05_kd': 'Expansion 5% + short teacher KD recovery'}

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
        timing=read(f"{folder}/{name}_t4.json", {})
        knobs={k:v for k,v in recipe.items() if k in ["pruning_ratio","round_to","pruning_scope","quantization","recovery"]}
        if knobs.get("recovery"):
            knobs["recovery"]={k:v for k,v in knobs["recovery"].items() if k != "train_indices"}
        rows.append({"name":name,"label":LABELS.get(name,name),"phase":phase,"status":"completed",
                     "accuracy_pct":accuracy["accuracy"]*100,"worst_recall_pct":accuracy["worst_class_accuracy"]*100,
                     "examples":accuracy["examples"],"scope":accuracy.get("evaluation_scope","historical_split_subset"),
                     "parameters":recipe.get("parameters"),"size_mib":model.stat().st_size/2**20 if model.exists() else None,
                     "median_cpu_4t_ms":timing.get("median_ms"),"estimated_cpu_4t_j":timing.get("energy_j_per_inference"),
                     "model_sha256":accuracy["model_sha256"],"settings":knobs,"evidence":evidence,
                     "implementation":["edgeprep/challenge.py","edgeprep/export.py","edgeprep/benchmark.py"]})
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
        failures=[v for v in jobs.values() if isinstance(v,dict) and v.get("code")=="FAILED"]
        status="failed" if failures else "completed" if profile and check else "pending"
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
                     "accuracy_pct":None,"energy_j":None,"failure_details":failures,
                     "settings":request,"jobs":jobs,"evidence":folder,
                     "implementation":["edgeprep/hub_profile.py","edgeprep/hub_auth.py"]})
    return rows


def table(rows, remote=False):
    if remote:
        fields=["name","status","device","precision","hub_estimated_latency_ms","top1_agreement_pct"]
    else:
        fields=["label","accuracy_pct","worst_recall_pct","size_mib","median_cpu_4t_ms","examples"]
    return pd.DataFrame([{k:r.get(k) for k in fields} for r in rows]).round(4)


def main():
    OUT.mkdir(exist_ok=True)
    historical=local_records("results/challenge", ["baseline","qdq_s8","qdq_u8","qdq_s8_256","qdq_percentile","qdq_protected","qdq_pointwise","prune05","prune10","expansion05","expansion10","expansion05_ce","expansion05_kd"], "historical_pilot")
    names=["baseline","qdq_pointwise","qdq_percentile512"]
    discovered=sorted((ROOT/"results/all_data").glob("*_accuracy.json"),key=lambda p:p.stat().st_mtime)
    names += [p.name.removesuffix("_accuracy.json") for p in discovered if p.name.removesuffix("_accuracy.json") not in names]
    current=local_records("results/all_data",names,"current_all_data")
    remote=hub_records()
    stages=[
      ("01_original_models", "Original models and dataset", "Verified supplied MobileNetV2 and optional ResNet50, exact safe checkpoint loading, fixed 224 input, all 27,000 EuroSAT files and preprocessing. MobileNetV2 has 2,236,682 parameters; ResNet50 is the teacher, never the denominator.", ["results/checkpoint_audit.json","results/dataset_audit.json","edgeprep/challenge_data.py"], []),
      ("02_historical_pilots", "Historical compression and short recovery", "Before the organizer clarification, 13 variants were screened on 500 images from a provisional split. The table follows the experimental progression, not reconstructed timestamps. Full quantization, selective quantization, physical pruning, and short CE/KD recovery are different operations. CE/KD used only 500 training images for one epoch. These rows cannot be mixed with current all-data ratios.",["docs/historical_pilot_analysis.md","results/challenge/summary.json"],historical),
      ("03_measured_baseline", "Our current unchanged-student baseline", "After the organizer clarified all-data training and our own laptop reference, evaluated the unchanged student on all 27,000 images. This is training-set accuracy. Exported FP32 ONNX and measured serial CPU runtime/RAM/estimated energy. No recovery or retraining.",["baseline_metrics.json","scripts/measure_all_data.ps1"],current[:1]),
      ("04_selective_int8", "Selective INT8 with 256 MinMax samples", "Quantized pointwise convolutions while preserving first/last layers, classifier and depthwise convolutions in FP32. Same learned parameters; smaller weight storage. Calibrated 256 balanced samples and evaluated all 27,000. Smaller does not mean faster: this graph regressed on our laptop CPU.",["ANALYSIS.md","edgeprep/export.py"],[r for r in current if r["name"]=="qdq_pointwise"]),
      ("05_percentile512", "Selective INT8 with 512 Percentile samples", "Same selective precision pattern, now Percentile calibration and 512 balanced samples. This changes two knobs together; it screens a configuration but does not isolate the benefit of either algorithm or sample count. Full-data evaluation and fresh runtime measurements recorded when present.",["results/all_data/calibration512/manifest.json","edgeprep/challenge.py"],[r for r in current if r["name"]=="qdq_percentile512"]),
      ("06_ai_hub", "Real-device CPU/NPU deployment", "Authenticated using the ignored .env key. Retrieved 78 catalogue entries. S24 CPU and NPU exploration, plus RB3 Hub-native W8A8 compilation. First S24 NPU inference failed because compute_unit and an explicit provider were both specified; the retry removes compute_unit from inference options. Remote 32-image agreement is a smoke check, never full-data accuracy. Preserve failures alongside successful runs.",["docs/ai_hub.md","results/hub/devices.json","edgeprep/hub_profile.py"],remote),
      ("07_next_experiments", "Next experiments, not yet measured", "Prioritize same-device W8A8 vs W8A16, fixed-count MSE vs MinMax calibration, beta Lite-MP, 1-2% expansion pruning with full-data CE/KD, and BN recalibration. Compare every candidate against a reference on the same device/runtime. No full-data recovery training has been completed yet.",["ANALYSIS.md","docs/ai_hub.md"],[])
    ]
    registry={"updated_utc":datetime.now(timezone.utc).isoformat(),"ordering":"Numbered phases and logical experimental progression; original wall-clock timestamps are not reconstructed", "records":historical+current+remote}
    (OUT/"runs.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    introduction="""# Experiment log â€” start here

Read the numbered folders in order. Each README explains **what changed, implementation, results and interpretation**, with links to the original evidence. `Experiment_Log.ipynb` presents the same tables with saved outputs; no rerunning training is required to read it. `runs.json` is the machine-readable registry.

Quantization reduces numerical precision and weight storage. Channel pruning physically removes parts of the network. Recovery trains a compressed model to restore accuracy. Device compilation prepares it for a particular runtime.

Two accuracy protocols exist: the historical 500-image pilot and the current 27,000-image training-set evaluation. They are separate tables. Remote device inference has a 32-image numerical check, not a new full-dataset accuracy score. All model changes still start from the supplied student.

"""
    index=introduction+"| Step | Experiment | Read |\n|---|---|---|\n"
    notebook_cells=[]
    def markdown(text):
        notebook_cells.append({"cell_type":"markdown","id":f"cell-{len(notebook_cells)}","metadata":{},"source":text})
    def code(source, html):
        notebook_cells.append({"cell_type":"code","id":f"cell-{len(notebook_cells)}","metadata":{},"source":source,"execution_count":len([c for c in notebook_cells if c["cell_type"]=="code"])+1,"outputs":[{"output_type":"display_data","data":{"text/html":html},"metadata":{}}]})
    markdown(introduction.replace("# Experiment log — start here", "# Sequential experiment notebook"))
    code('from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display\nproject = Path.cwd()\nif not (project / "edgeprep").exists(): project = project.parent\nregistry = json.loads((project / "experiments/runs.json").read_text())\ndisplay(pd.DataFrame([{k:r.get(k) for k in ["name","phase","status"]} for r in registry["records"]]))',pd.DataFrame([{k:r.get(k) for k in ["name","phase","status"]} for r in registry["records"]]).to_html(index=False))
    for n,(slug,title,explanation,evidence,rows) in enumerate(stages,1):
        folder=OUT/slug;folder.mkdir(exist_ok=True)
        index+=f"| {n:02} | {title} | [Details]({slug}/README.md) |\n"
        body=f"# {n:02}. {title}\n\n{explanation}\n\n"
        markdown(body)
        if rows:
            df=table(rows,n==6)
            headers=list(df.columns)
            rendered="| "+" | ".join(headers)+" |\n|"+"|".join("---" for _ in headers)+"|\n"
            for values in df.itertuples(index=False,name=None):
                rendered+="| "+" | ".join("pending / n/a" if pd.isna(v) else str(v) for v in values)+" |\n"
            body+=rendered+"\n"
            (folder/"details.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            phase="remote_exploration" if n==6 else "historical_pilot" if n==2 else "current_all_data"
            names=[r["name"] for r in rows];fields=list(df.columns)
            source=f'rows = [r for r in registry["records"] if r["phase"] == {phase!r} and r["name"] in {names!r}]\ndisplay(pd.DataFrame([{{k:r.get(k) for k in {fields!r}}} for r in rows]).round(4))'
            code(source,df.to_html(index=False,na_rep="pending / n/a"))
            if n == 6:
                body += "All latency values in this table are Hub estimates. S24 CPU/NPU comparisons use the same source; numerical agreement is only 32 images. Actual placement, memory ranges, failure reasons and job IDs are in details.json.\n\n"
            body+="## Implementation and exact settings\n\nOpen [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.\n\n"
            for r in rows:
                body+=f"- **{r.get('label',r['name'])}** (`{r['name']}`): {linked(r['evidence'])}; "+", ".join(linked(x) for x in r["implementation"])+".\n"
        body+="\n## Supporting evidence\n\n"+"\n".join("- "+linked(x) for x in evidence)+"\n"
        (folder/"README.md").write_text(body,encoding="utf-8")
    index+="\nCurrent CPU results:\n\n"+table(current).to_html(index=False,na_rep="pending / n/a")+"\n\nTo refresh this index after more experiments (reads results only):\n\n```powershell\n.venv\\Scripts\\python.exe -m edgeprep.experiment_log\n```\n"
    index += "\nOpen the notebook interactively:\n\n```powershell\npowershell -ExecutionPolicy Bypass -File scripts/notebook.ps1\n```\n\nJupyterLab and the project kernel are installed locally. The notebook reads evidence only; it never reruns training or submits Hub jobs. Timing JSON files also contain a one-image smoke-check field called accuracy; use the separate *_accuracy.json dataset measurements in these tables.\n"
    (OUT/"README.md").write_text(index,encoding="utf-8")
    plot=ROOT/"results/all_data/tradeoff.png"
    if plot.exists():
        markdown("## Current all-data tradeoff\nTraining-set accuracy; same laptop and input contract. Remote device results stay separate.")
        notebook_cells.append({"cell_type":"code","id":"tradeoff","metadata":{},"source":'from IPython.display import Image\ndisplay(Image(filename=str(project / "results/all_data/tradeoff.png")))',"execution_count":len([c for c in notebook_cells if c["cell_type"]=="code"])+1,"outputs":[{"output_type":"display_data","data":{"image/png":base64.b64encode(plot.read_bytes()).decode()},"metadata":{}}]})
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3 (project .venv)","language":"python","name":"edgeprep"},"language_info":{"name":"python","version":"3.11"}},"cells":notebook_cells}
    (OUT/"Experiment_Log.ipynb").write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    print(f"Updated {OUT}: {len(registry['records'])} run entries and a notebook with saved outputs")


if __name__ == "__main__":main()
