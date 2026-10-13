"""EuroSAT competition tools. Original checkpoint -> reproducible structural recipe -> ONNX."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path, PureWindowsPath
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torch_pruning as tp
from torchvision import models
from thop import profile

from .challenge_data import dataset, data_protocol, balanced_ids, prepare, digest
from .export import export_fp32, quantize


def load_model(path, teacher=False):
    model = models.resnet50(weights=None) if teacher else models.mobilenet_v2(weights=None)
    if teacher:
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def prune_model(model, ratio, round_to=8, scope="uniform"):
    if not 0 <= ratio < .6:
        raise ValueError("Use a conservative pruning ratio in [0, .6)")
    if ratio == 0:
        return model
    torch.manual_seed(42)
    if scope == "expansion":
        # Remove only hidden expansion channels; preserve every bottleneck output and
        # residual width. Rank BN-scaled projection-column norms, not raw conv scale.
        dg = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))
        for block in model.features[1:18]:
            if len(block.conv) != 4:
                continue
            expansion, bn = block.conv[0][0], block.conv[0][1]
            projection = block.conv[2]
            width = expansion.out_channels
            kept = int(width*(1-ratio))
            if round_to:
                kept = max(round_to,(kept//round_to)*round_to)
            remove = width-kept
            if remove <= 0:
                continue
            score = bn.weight.detach().abs()*projection.weight.detach().square().sum((0,2,3)).sqrt()
            indices = torch.argsort(score,stable=True)[:remove].tolist()
            group = dg.get_pruning_group(expansion,tp.prune_conv_out_channels,idxs=indices)
            if not dg.check_pruning_group(group):
                raise RuntimeError("Unsafe expansion dependency group")
            group.prune()
        return model.eval()
    # Protect input/output features and the ten-class head. DepGraph handles residual
    # additions and changes depthwise groups alongside channels. No mask-only pruning.
    protected = [model.features[0], model.features[18], model.classifier]
    pruner = tp.pruner.MagnitudePruner(model, torch.randn(1,3,224,224),
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio, ignored_layers=protected, round_to=round_to,
        root_module_types=[nn.Conv2d], global_pruning=False)
    pruner.step()
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) and layer.groups > 1:
            if layer.in_channels != layer.out_channels or layer.groups != layer.in_channels:
                raise RuntimeError("Invalid depthwise channel/group dependency after pruning")
    return model.eval()


def restore_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    source = Path(recipe["source_checkpoint"])
    if not source.exists():
        source = Path("Models") / PureWindowsPath(recipe["source_checkpoint"]).name
    if digest(source) != recipe["source_sha256"]:
        raise ValueError("Checkpoint differs from the recipe's original source")
    model = prune_model(load_model(source), recipe["pruning_ratio"], recipe["round_to"],recipe.get("pruning_scope","uniform"))
    if recipe.get("internal_resolution"):
        from .resolution import InternalResize
        model = InternalResize(model, recipe["internal_resolution"]).eval()
    if recipe.get("recovered_checkpoint"):
        recovered = Path(recipe["recovered_checkpoint"])
        if not recovered.exists(): recovered = Path(path).parent / "recovered_state.pt"
        if recipe.get("recovered_sha256") and digest(recovered)!=recipe["recovered_sha256"]:
            raise ValueError("Recovered checkpoint hash differs from recipe")
        state = torch.load(recovered, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model, recipe


def export_candidate(args):
    if args.recipe:
        model, recipe = restore_recipe(args.recipe)
    else:
        model = prune_model(load_model(args.checkpoint), args.prune, args.round_to,args.pruning_scope)
        recipe = {"source_checkpoint": str(Path(args.checkpoint).resolve()),
                  "source_sha256": digest(args.checkpoint), "pruning_ratio": args.prune,
                  "round_to": args.round_to, "pruning_scope":args.pruning_scope,
                  "seed": 42, "torch_pruning_version": importlib.metadata.version("torch-pruning")}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    macs, _ = profile(copy.deepcopy(model), inputs=(torch.randn(1,3,224,224),), verbose=False)
    error = export_fp32(model, out / "model_fp32.onnx", size=224, opset=13, fixed_batch=True)
    recipe.update(parameters=params, macs=macs, input_shape=[1,3,224,224], opset=13,
                  fp32_parity_max_error=error, quantization=None)
    if args.calibration:
        arrays = np.load(args.calibration, mmap_mode="r", allow_pickle=False)
        if arrays.ndim != 4 or arrays.shape[1:] != (3,224,224) or not len(arrays):
            raise ValueError("Calibration requires nonempty N x 3 x 224 x 224 float inputs")
        count = min(len(arrays), args.calibration_count)
        if count < 1 or not np.isfinite(arrays[:count]).all():
            raise ValueError("Calibration is empty or nonfinite")
        excluded = []
        if args.protect_ends:
            graph = onnx.load(str(out / "model_fp32.onnx"))
            excluded = [n.name for n in graph.graph.node if n.op_type in ("Conv", "Gemm", "MatMul")
                        and any(key in n.name for key in ("features.0/", "features.18/", "classifier/"))]
            if not excluded:
                raise ValueError("No end nodes found; inspect exported graph names")
        if args.protect_depthwise:
            graph = onnx.load(str(out / "model_fp32.onnx"))
            excluded += [n.name for n in graph.graph.node if n.op_type=="Conv"
                         and any(a.name=="group" and a.i>1 for a in n.attribute)]
        quantize(out / "model_fp32.onnx", out / "model_int8.onnx",
                 (arrays[i:i+1] for i in range(count)), unsigned=args.unsigned,
                 activation_symmetric=False, nodes_to_exclude=excluded, method=args.method)
        recipe["quantization"] = {"format": "static QDQ", "method": args.method,
            "types": "U8U8" if args.unsigned else "S8S8", "activation_symmetric": False,
            "per_channel": True, "calibration_count": count, "excluded_nodes": excluded,
            "calibration_sha256": digest(args.calibration)}
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    summary=copy.deepcopy(recipe)
    if summary.get("recovery"):
        ids=summary["recovery"].pop("train_indices",None)
        summary["recovery"]["training_examples"]=len(ids) if ids is not None else None
    print(json.dumps(summary,indent=2))


def evaluate(args):
    ds = dataset(args.root)
    protocol = data_protocol(len(ds), args.split)
    ids = balanced_ids(protocol["evaluation"], ds.targets, args.limit)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])
    entry = session.get_inputs()[0]
    if entry.shape != [1,3,224,224]:
        raise ValueError(f"Input contract mismatch: {entry.shape}")
    matrix = np.zeros((10,10),dtype=np.int64)
    started = time.perf_counter()
    for position, idx in enumerate(ids, 1):
        x,y = ds[idx]
        logits = session.run(None,{entry.name:x.numpy()[None]})[0]
        if logits.shape != (1,10) or not np.isfinite(logits).all():
            raise ValueError("Invalid class output")
        matrix[y,int(logits.argmax())] += 1
        if position % 2000 == 0:
            print(f"Evaluated {position}/{len(ids)} images", flush=True)
    support = matrix.sum(1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(10,dtype=float), where=support>0)
    result = {"model": str(Path(args.model).resolve()), "model_sha256": digest(args.model),
              "accuracy": float(matrix.trace()/matrix.sum()), "examples": len(ids),
              "per_class_recall": dict(zip(ds.classes,recalls.tolist())), "class_support": support.tolist(),
              "worst_class_accuracy": float(recalls[support>0].min()), "confusion_matrix":matrix.tolist(),
              "evaluation_indices": ids, "split_sha256": protocol["split_sha256"],
              "evaluation_indices_sha256": __import__("hashlib").sha256(np.array(ids,dtype="<i8").tobytes()).hexdigest(),
              "protocol": protocol["protocol"], "evaluation_scope": protocol["evaluation_scope"],
              "provisional": protocol["provisional"], "full_evaluation_count":len(protocol["evaluation"]),
              "evaluation_seconds":time.perf_counter()-started,
              "accuracy_caveat":protocol["accuracy_caveat"],
              "threads":args.threads,"provider":"CPUExecutionProvider"}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("evaluation_indices","confusion_matrix")},indent=2))


def training_device(requested):
    """Never silently fall back to CPU when GPU training was requested."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires .venv-train-gpu. Run scripts/setup_gpu_training.ps1, then use .venv-train-gpu/Scripts/python.exe. Explicit --device cpu is for historical reproduction only.")
    return torch.device(requested)


def recover(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.train_limit < 0 or args.temperature <= 0 or not 0 <= args.alpha <= 1 or args.workers < 0:
        raise ValueError("Invalid recovery epochs, batch size, rate, limit, temperature, alpha or workers")
    device = training_device(args.device)
    amp = args.amp and device.type == "cuda"
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    execution = {"device":str(device), "device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                 "torch_version":torch.__version__, "cuda_version":torch.version.cuda, "amp":amp,
                 "amp_dtype":"float16" if amp else None, "workers":args.workers,
                 "tf32":False, "cudnn_benchmark":False, "checkpoint_tensor_device":"cpu"}
    print(json.dumps({"training_execution":execution}),flush=True)
    model, recipe = restore_recipe(args.recipe)
    model = model.to(device)
    if args.head_only:
        model.requires_grad_(False)
        backbone = model.student if hasattr(model, "student") else model
        backbone.classifier.requires_grad_(True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ds = dataset(args.root)
    protocol = data_protocol(len(ds), args.split)
    train_ids = balanced_ids(protocol["train"],ds.targets,args.train_limit)
    teacher = load_model(args.teacher,teacher=True).to(device).requires_grad_(False) if args.teacher else None
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds,train_ids),
                                        batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                                        generator=torch.Generator().manual_seed(42))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter();history=[]
    for epoch in range(args.epochs):
        model.train();total_loss=0.;count=0
        # Freeze BN running statistics for short, small-batch recovery.
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d): layer.eval()
        last_report=0
        for x,y in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits=model(x);loss=nn.functional.cross_entropy(logits,y)
                if teacher is not None:
                    with torch.no_grad(): target=teacher(x)
                    # Distillation probabilities use FP32 even with mixed-precision convolution.
                    soft=nn.functional.kl_div(nn.functional.log_softmax(logits.float()/args.temperature,dim=1),
                        nn.functional.softmax(target.float()/args.temperature,dim=1),reduction="batchmean")*args.temperature**2
                    loss=(1-args.alpha)*loss+args.alpha*soft
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite recovery loss")
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=loss.item()*len(y);count+=len(y)
            if count-last_report >= 1000 or count==len(train_ids):
                progress={"status":"running","epoch":epoch+1,"epochs":args.epochs,"samples_this_epoch":count,
                    "training_examples":len(train_ids),"elapsed_s":time.perf_counter()-started,"loss":total_loss/count,
                    "head_only":args.head_only,"scope":protocol["protocol"],"execution":execution}
                (out/"training_progress.json").write_text(json.dumps(progress,indent=2),encoding="utf-8")
                print(f"Recovery epoch {epoch+1}: {count}/{len(train_ids)} samples, loss={total_loss/count:.4f}",flush=True)
                last_report=count
        history.append({"epoch":epoch+1,"loss":total_loss/count})
        print(history[-1],flush=True)
    if device.type == "cuda": torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    # GPU training artifacts remain loadable and exportable in the CPU environment.
    torch.save({k:v.detach().cpu() for k,v in model.eval().state_dict().items()},out/"recovered_state.pt")
    recipe.update(training="Classifier-only recovery; backbone frozen" if args.head_only else "Full-backbone recovery", recovered_checkpoint=str((out/"recovered_state.pt").resolve()),
                  recovered_sha256=digest(out/"recovered_state.pt"),
                  recovery={"epochs":args.epochs,"train_limit":args.train_limit,"train_indices":train_ids,
                            "teacher":str(Path(args.teacher).resolve()) if args.teacher else None,
                            "teacher_sha256":digest(args.teacher) if args.teacher else None,
                            "alpha":args.alpha,"temperature":args.temperature,"lr":args.lr,
                            "seed":42,"batch_size":args.batch_size,"bn_running_stats":"frozen","head_only":args.head_only,
                            "trained_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "split_sha256":protocol["split_sha256"], "protocol":protocol["protocol"],
                            "available_train_count":len(protocol["train"]), "history":history,
                            "seconds":seconds,"execution":execution})
    (out/"recipe.json").write_text(json.dumps(recipe,indent=2),encoding="utf-8")
    progress.update(status="completed",elapsed_s=seconds)
    (out/"training_progress.json").write_text(json.dumps(progress,indent=2),encoding="utf-8")


def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("prepare");d.add_argument("--root",default="data");d.add_argument("--split")
    d.add_argument("--output",default="results/all_data/challenge_data");d.add_argument("--calibration-count",type=int,default=256)
    e=sub.add_parser("export");e.add_argument("--checkpoint",default="Models/student_mobilenetv2_distilled.pt")
    e.add_argument("--recipe");e.add_argument("--output",required=True);e.add_argument("--prune",type=float,default=0)
    e.add_argument("--round-to",type=int,default=8);e.add_argument("--calibration")
    e.add_argument("--pruning-scope",choices=["uniform","expansion"],default="expansion")
    e.add_argument("--calibration-count",type=int,default=256);e.add_argument("--unsigned",action="store_true")
    e.add_argument("--protect-ends",action="store_true");e.add_argument("--method",choices=["MinMax","Entropy","Percentile"],default="MinMax")
    e.add_argument("--protect-depthwise",action="store_true")
    v=sub.add_parser("evaluate");v.add_argument("--model",required=True);v.add_argument("--root",default="data")
    v.add_argument("--split",help="Optional legacy split; omitted uses every EuroSAT image");v.add_argument("--output",required=True)
    v.add_argument("--limit",type=int,default=0,help="0 evaluates all data; nonzero is a labelled diagnostic subset")
    v.add_argument("--threads",type=int,default=1)
    f=sub.add_parser("recover");f.add_argument("--recipe",required=True);f.add_argument("--split",help="Optional legacy split; omitted trains on all data")
    f.add_argument("--root",default="data");f.add_argument("--teacher");f.add_argument("--output",required=True)
    f.add_argument("--epochs",type=int,default=1);f.add_argument("--train-limit",type=int,default=0,help="0 uses all 27000 images; nonzero is a short diagnostic run")
    f.add_argument("--device",choices=["cuda","cpu"],default="cuda",help="CUDA by default; CPU only for explicit historical reproduction")
    f.add_argument("--workers",type=int,default=0)
    f.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True,help="CUDA mixed precision; disabled on CPU")
    f.add_argument("--head-only",action="store_true",help="Freeze backbone; only adapt the ten-class classifier")
    f.add_argument("--batch-size",type=int,default=8);f.add_argument("--lr",type=float,default=1e-4)
    f.add_argument("--temperature",type=float,default=4);f.add_argument("--alpha",type=float,default=.5)
    args=p.parse_args()
    if args.command=="prepare":print(json.dumps(prepare(args.root,args.split,args.output,args.calibration_count),indent=2))
    elif args.command=="export":export_candidate(args)
    elif args.command=="evaluate":evaluate(args)
    else:recover(args)


if __name__=="__main__":main()
