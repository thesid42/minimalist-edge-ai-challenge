"""EuroSAT all-data protocol; legacy disjoint splits remain available for historic experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset(root="data", download=False):
    transform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return datasets.EuroSAT(root=root, download=download, transform=transform)


def validate_split(split, count):
    if not isinstance(split, dict) or not all(k in split for k in ("train", "val")):
        raise ValueError("Split must contain train and val index lists")
    for key in ("train", "val"):
        ids = split[key]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{key} must be a nonempty list")
        if any(type(i) is not int or i < 0 or i >= count for i in ids):
            raise ValueError(f"Invalid {key} dataset index")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {key} indices")
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Training and validation overlap")
    return split


def load_split(path, count):
    return validate_split(json.loads(Path(path).read_text(encoding="utf-8")), count)


def data_protocol(count, split_path=None):
    if split_path:
        split = load_split(split_path, count)
        return {"train": split["train"], "evaluation": split["val"],
                "protocol": "legacy_split", "provisional": split.get("provisional", False),
                "evaluation_scope": "split_validation", "split_sha256": digest(split_path),
                "accuracy_caveat": "Historical split experiment; checkpoint training membership unknown"}
    ids = list(range(count))
    return {"train": ids, "evaluation": ids.copy(), "protocol": "organizer_all_data",
            "provisional": False, "evaluation_scope": "training_set",
            "split_sha256": None,
            "accuracy_caveat": "Training-set accuracy under organizer's all-data protocol; no held-out generalization estimate"}


def balanced_ids(ids, targets, count, seed=42):
    """Interleave shuffled classes; return only members of the supplied partition."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation([i for i in ids if targets[i] == c]).tolist() for c in range(10)]
    output = []
    while any(groups) and (count == 0 or len(output) < count):
        for group in groups:
            if group and (count == 0 or len(output) < count):
                output.append(group.pop())
    return output


def prepare(root="data", split_path=None, output="results/all_data/challenge_data", calibration_count=256):
    ds = dataset(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = data_protocol(len(ds), split_path)
    status = "organizer_all_data" if split_path is None else "legacy_split"
    calib_ids = balanced_ids(protocol["train"], ds.targets, calibration_count)
    calibration = np.stack([ds[i][0].numpy() for i in calib_ids])
    np.save(out / "calibration.npy", calibration)
    first = protocol["evaluation"][0]
    np.save(out / "timing_input.npy", ds[first][0].numpy()[None])
    np.save(out / "timing_label.npy", np.array([ds.targets[first]]))
    samples = [{"path": str(Path(p).relative_to(Path(ds._data_folder))), "label": y} for p,y in ds.samples]
    ordering = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
    metadata = {"dataset": "EuroSAT RGB", "examples": len(ds), "class_to_idx": ds.class_to_idx,
                "protocol": protocol["protocol"], "split_status": status,
                "split_path": str(Path(split_path).resolve()) if split_path else None,
                "split_sha256": protocol["split_sha256"], "sample_order_sha256": ordering,
                "train_count": len(protocol["train"]), "evaluation_count": len(protocol["evaluation"]),
                "evaluation_scope": protocol["evaluation_scope"],
                "calibration_indices": calib_ids, "calibration_count": len(calib_ids),
                "calibration_sha256": digest(out / "calibration.npy"),
                "input_shape": [1,3,224,224], "mean": [.485,.456,.406], "std": [.229,.224,.225],
                "preprocessing": "torchvision Resize(224) bilinear default antialias; ToTensor; ImageNet Normalize",
                "held_out_split": split_path is not None,
                "calibration_overlaps_evaluation": split_path is None,
                "accuracy_caveat": protocol["accuracy_caveat"]}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
