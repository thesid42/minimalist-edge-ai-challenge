"""Exact EuroSAT preprocessing, organizer split validation, labelled provisional fallback."""
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


def prepare(root="data", split_path=None, output="results/challenge_data", calibration_count=256):
    ds = dataset(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if split_path:
        split = load_split(split_path, len(ds))
        status = "organizer_split_supplied_by_user"
    else:
        # This does NOT reconstruct the checkpoint's training split. Its accuracy can leak.
        rng = np.random.default_rng(42)
        split = {"train": [], "val": [], "provisional": True,
                 "warning": "NOT organizer split; checkpoint training overlap is unknown"}
        for c in range(10):
            ids = rng.permutation(np.flatnonzero(np.array(ds.targets) == c)).tolist()
            nval = len(ids) // 5
            split["val"].extend(ids[:nval])
            split["train"].extend(ids[nval:])
        split_path = out / "provisional_split.json"
        Path(split_path).write_text(json.dumps(split, indent=2), encoding="utf-8")
        status = "provisional_not_official"
    validate_split(split, len(ds))
    calib_ids = balanced_ids(split["train"], ds.targets, calibration_count)
    calibration = np.stack([ds[i][0].numpy() for i in calib_ids])
    np.save(out / "calibration.npy", calibration)
    np.save(out / "timing_input.npy", ds[split["val"][0]][0].numpy()[None])
    np.save(out / "timing_label.npy", np.array([ds.targets[split["val"][0]]]))
    samples = [{"path": str(Path(p).relative_to(Path(ds._data_folder))), "label": y} for p,y in ds.samples]
    ordering = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
    metadata = {"dataset": "EuroSAT RGB", "examples": len(ds), "class_to_idx": ds.class_to_idx,
                "split_status": status, "split_path": str(Path(split_path).resolve()),
                "split_sha256": digest(split_path), "sample_order_sha256": ordering,
                "train_count": len(split["train"]), "val_count": len(split["val"]),
                "calibration_indices": calib_ids, "calibration_count": len(calib_ids),
                "calibration_sha256": digest(out / "calibration.npy"),
                "input_shape": [1,3,224,224], "mean": [.485,.456,.406], "std": [.229,.224,.225],
                "preprocessing": "torchvision Resize(224) bilinear default antialias; ToTensor; ImageNet Normalize",
                "validation_used_for_calibration": False,
                "accuracy_caveat": "Provisional accuracy is diagnostic only; supplied checkpoint training membership unknown"}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
