# 01. Original models and dataset

Verified supplied MobileNetV2 and optional ResNet50, exact safe checkpoint loading, fixed 224 input, all 27,000 EuroSAT files and preprocessing. MobileNetV2 has 2,236,682 parameters; ResNet50 is the teacher, never the denominator.


## Supporting evidence

- [results/checkpoint_audit.json](../../results/checkpoint_audit.json)
- [results/dataset_audit.json](../../results/dataset_audit.json)
- [edgeprep/challenge_data.py](../../edgeprep/challenge_data.py)
