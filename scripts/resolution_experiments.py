# Full-data evaluation of two spatial-compute candidates; independent of remote jobs.
from pathlib import Path
import subprocess
import sys
for size in [192,160]:
    out=f'artifacts/all_data/resize{size}'
    if not Path(out+'/model_fp32.onnx').exists():
        subprocess.run([sys.executable,'-m','edgeprep.resolution','--size',str(size),'--output',out],check=True)
    result=f'results/all_data/resize{size}_accuracy.json'
    if not Path(result).exists():
        subprocess.run([sys.executable,'-m','edgeprep.challenge','evaluate','--model',out+'/model_fp32.onnx','--threads','4','--output',result],check=True)
