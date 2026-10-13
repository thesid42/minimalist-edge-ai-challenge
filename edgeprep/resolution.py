"""Keep the required 224 input, resize inside the graph, then run the supplied student."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from thop import profile
from .challenge import load_model
from .challenge_data import digest
from .export import export_fp32

class InternalResize(nn.Module):
    def __init__(self,student,size):
        super().__init__();self.student=student;self.size=size
    def forward(self,images):
        x=nn.functional.interpolate(images,size=(self.size,self.size),mode='bilinear',align_corners=False)
        return self.student(x)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint',default='Models/student_mobilenetv2_distilled.pt')
    a=p.parse_args()
    if a.size not in (128,160,192,224):p.error('Use a conservative supported spatial size')
    torch.set_num_threads(4);torch.manual_seed(42)
    student=load_model(a.checkpoint)
    macs,_=profile(student,inputs=(torch.zeros(1,3,a.size,a.size),),verbose=False)
    model=InternalResize(student,a.size).eval();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    error=export_fp32(model,out/'model_fp32.onnx',size=224,opset=13,fixed_batch=True)
    recipe={'source_checkpoint':str(Path(a.checkpoint).resolve()),'source_sha256':digest(a.checkpoint),
      'pruning_ratio':0,'pruning_scope':'expansion','round_to':8,'quantization':None,
      'parameters':sum(p.numel() for p in student.parameters()),'macs':macs,
      'macs_note':'THOP convolution/linear MACs at internal size; bilinear Resize cost excluded',
      'internal_resolution':a.size,'input_shape':[1,3,224,224],'opset':13,'fp32_parity_max_error':error,
      'resize':'Inside exported graph, bilinear align_corners=False, antialias=False, normalized input',
      'training':'None; original learned weights unchanged',
      'challenge_reference':'PDF section 3 allows lower resolution but fixes external (1,3,224,224) input'}
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8');print(json.dumps(recipe,indent=2))
if __name__=='__main__':main()
