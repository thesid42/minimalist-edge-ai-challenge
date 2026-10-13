"""Physical FP32 architecture edits, replayed before loading the saved state."""
import torch
from torch import nn
import torch_pruning as tp


def apply_edits(model, edits, size=192):
    backbone = model.student if hasattr(model, "student") else model
    for edit in edits:
        if edit["kind"] == "drop_residual":
            index = edit["feature_index"]
            block = backbone.features[index]
            if not getattr(block, "use_res_connect", False):
                raise ValueError("Only shape-preserving residual blocks can be removed")
            backbone.features[index] = nn.Identity()
        elif edit["kind"] == "prune_conv_out":
            layer = backbone.get_submodule(edit["module"])
            indices = edit["indices"]
            if not isinstance(layer, nn.Conv2d) or not indices or len(indices) != len(set(indices)):
                raise ValueError("Invalid structural channel removal")
            if min(indices) < 0 or max(indices) >= layer.out_channels or len(indices) >= layer.out_channels:
                raise ValueError("Invalid pruning channel indices")
            # DepGraph needs autograd tracing even though this is a CPU export operation.
            with torch.enable_grad():
                graph = tp.DependencyGraph().build_dependency(backbone, example_inputs=torch.zeros(1, 3, size, size))
                group = graph.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                if not graph.check_pruning_group(group):
                    raise ValueError("Unsafe channel dependency group")
                group.prune()
        else:
            raise ValueError("Unknown structural edit")
    for layer in backbone.modules():
        if isinstance(layer, nn.Conv2d) and layer.groups > 1:
            if layer.in_channels != layer.out_channels or layer.groups != layer.in_channels:
                raise ValueError("Broken depthwise channel dependency")
    if backbone.classifier[1].out_features != 10:
        raise ValueError("Ten-class output contract changed")
    return model.eval()


def ranked_indices(score, count):
    if count < 1 or count >= score.numel():
        raise ValueError("Invalid requested channel reduction")
    return torch.argsort(score, stable=True)[:count].tolist()


def plan_edits(model, mode):
    b = model.student if hasattr(model, "student") else model
    if mode == "head640":
        # Keep 640 of 1280 final channels; DepGraph also slices BN and classifier inputs.
        score = b.features[18][1].weight.detach().abs() * b.classifier[1].weight.detach().square().sum(0).sqrt()
        return [{"kind": "prune_conv_out", "module": "features.18.0", "indices": ranked_indices(score, b.features[18][0].out_channels - 640)}]
    if mode == "late_expansion10":
        edits = []
        # Protect all early stages and every residual/output width.
        for index in range(14, 18):
            block = b.features[index]
            width = block.conv[0][0].out_channels
            keep = int(width * .90) // 8 * 8
            score = block.conv[0][1].weight.detach().abs() * block.conv[2].weight.detach().square().sum((0, 2, 3)).sqrt()
            edits.append({"kind": "prune_conv_out", "module": f"features.{index}.conv.0.0", "indices": ranked_indices(score, width - keep)})
        return edits
    if mode == "drop16":
        return [{"kind": "drop_residual", "feature_index": 16}]
    raise ValueError("Unknown compression mode")
