"""Generate the challenge Section 8 metrics.json for the selected FP32 model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('accuracy','params_million','size_mb','macs_giga','latency_ms','peak_ram_mb','joules_per_inference')

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--team-name',required=True)
 args=parser.parse_args()
 if not args.team_name.strip():raise ValueError('Team name is required')
 current=read('best_models/current.json')
 bundle=Path(current['bundle'])
 manifest=read(bundle/'manifest.json')
 recipe=read(bundle/'recipe.json')
 accuracy=read('results/fp32_compression/head640_head_ce_accuracy.json')
 baseline_accuracy=read('results/all_data/baseline_accuracy.json')
 rows={r['name']:r for r in read('results/fp32_compression/summary.json')['records']}
 runtime={r['name']:r for r in read('results/fp32_compression/runtime/summary.json')['records']}
 final_runtime=runtime['head640_head_ce'];base_runtime=runtime['original224_fp32']
 model=ROOT/bundle/manifest['inference_model']
 assert current['name']=='resize192_head640_head_ce_fp32'
 assert hashlib.sha256(model.read_bytes()).hexdigest()==current['model_sha256']==accuracy['model_sha256']==final_runtime['sha256']
 assert baseline_accuracy['model_sha256']==base_runtime['sha256']==rows['original224_fp32']['model_sha256']
 assert accuracy['examples']==baseline_accuracy['examples']==27000
 assert accuracy['evaluation_indices_sha256']==baseline_accuracy['evaluation_indices_sha256']
 assert manifest['weight_precision']==manifest['cpu_compute_precision']=='FP32'
 assert accuracy['evaluation_scope']==baseline_accuracy['evaluation_scope']=='training_set'
 assert final_runtime['platform']==base_runtime['platform']
 assert final_runtime['codecarbon_record']['cpu_model']==base_runtime['codecarbon_record']['cpu_model']
 for runtime_row in (base_runtime,final_runtime):
  assert runtime_row['provider']=='CPUExecutionProvider' and runtime_row['threads']==4 and runtime_row['replicates']==3
  assert runtime_row['codecarbon_mode']=='tdp_estimate' and runtime_row['energy_gpu_ids']==[]
  raw=[read(p) for p in runtime_row['evidence']]
  assert all(r['sha256']==runtime_row['sha256'] and r['threads']==4 and r['runs']==1000 and r['warmup']==100 for r in raw)
  assert math.isclose(runtime_row['median_ms'],statistics.median(r['median_ms'] for r in raw))
  assert math.isclose(runtime_row['energy_j_per_inference'],statistics.mean(r['energy_j_per_inference'] for r in raw))
 def section(row,score,timing):
  result={'accuracy':score['accuracy'],'params_million':row['parameters']/1e6,'size_mb':timing['file_bytes']/2**20,
          'macs_giga':row['macs']/1e9,'latency_ms':timing['median_ms'],'peak_ram_mb':timing['peak_rss_mib'],
          'joules_per_inference':timing['energy_j_per_inference']}
  assert set(result)==set(FIELDS) and all(math.isfinite(v) and v>0 for v in result.values())
  assert 0<=result['accuracy']<=1
  return result
 baseline=section(rows['original224_fp32'],baseline_accuracy,base_runtime)
 final=section(rows['head640_head_ce'],accuracy,final_runtime)
 assert final['size_mb']==model.stat().st_size/2**20
 assert accuracy['worst_class_accuracy']==min(accuracy['per_class_recall'].values())
 record=final_runtime['codecarbon_record']
 report={
  'team_name':args.team_name.strip(),
  'baseline':baseline,
  'final':final,
  'worst_class_accuracy':accuracy['worst_class_accuracy'],
  'codecarbon_mode':'tdp_estimate',
  'techniques_applied':[
   'Internal resolution reduction to 192x192 with fixed external (1,3,224,224) input',
   'Structured final-feature channel pruning: 1280 to 640 channels, with dependent BN and classifier inputs removed',
   'Classifier-only cross-entropy recovery on all 27000 images; FP32 weights and compute'],
  'ai_hub_profile_summary':'No AI Hub compile/profile for this exact selected FP32 model. All submitted efficiency figures are laptop CPU measurements; older models\' Hub profiles are not attributed to this model.',
  'measurement_machine':f"{record['cpu_model']}; {record['ram_total_size']:.2f} GiB RAM; {final_runtime['platform']}; ONNX Runtime {final_runtime['onnxruntime']}, CPUExecutionProvider, 4 threads, batch 1",
  'measurement_details':{
   'schema_source':'The Minimalist Edge AI Challenge.pdf, Section 8',
   'baseline_role':'Unchanged supplied MobileNetV2, measured locally under organizer own-laptop-baseline clarification',
   'model_format':'Uncompressed FP32 ONNX for both baseline and final',
   'accuracy_units':'Fraction between 0 and 1',
   'size_and_ram_units':'MiB (2**20 bytes), matching the brief code despite its mb field labels',
   'evaluation_scope':'training_set',
   'training_examples':27000,'evaluation_examples':27000,
   'data_protocol':'Organizer clarification: train and evaluate on all EuroSAT images; no held-out split',
   'input_shape':[1,3,224,224],'output_shape':[1,10],'internal_resolution':192,
   'weight_precision':'FP32','cpu_compute_precision':'FP32','quantization':None,
   'latency_statistic':'Median of three per-run medians; 1000 timings after 100 warmups per fresh process',
   'latency_input':'One real preprocessed image repeated; accuracy measured separately on all 27000 images',
   'benchmark_order':'Serial fresh processes, rotating model order, no concurrent training',
   'peak_ram_statistic':'Maximum sampled process RSS across three repetitions',
   'energy_statistic':'Mean of three 30-second CodeCarbon windows',
   'energy_scope':'Process-attributed CPU TDP/load estimate plus estimated system RAM; GPU excluded; not a hardware meter reading',
   'codecarbon_version':record['codecarbon_version'],'cpu_tdp_w':final_runtime['codecarbon_cpu_tdp_w'],
   'energy_sample_sd_j':{'baseline':base_runtime['energy_stdev'],'final':final_runtime['energy_stdev']},
   'latency_range_ms':{'baseline':[base_runtime['median_min_ms'],base_runtime['median_max_ms']], 'final':[final_runtime['median_min_ms'],final_runtime['median_max_ms']]},
   'macs_note':'THOP computation counts exclude the internal bilinear resize overhead',
   'per_class_recall':{'baseline':baseline_accuracy['per_class_recall'],'final':accuracy['per_class_recall']},
   'baseline_worst_class_accuracy':baseline_accuracy['worst_class_accuracy'],
   'final_model':str((bundle/manifest['inference_model']).as_posix()),
   'baseline_model_sha256':base_runtime['sha256'],'final_model_sha256':current['model_sha256'],
   'final_checkpoint_sha256':manifest['files']['recovered_state.pt']['sha256'],
   'ratios_final_to_baseline':{field:final[field]/baseline[field] for field in FIELDS},
   'accuracy_change_percentage_points':(final['accuracy']-baseline['accuracy'])*100,
   'evidence':{
    'baseline_accuracy':'results/all_data/baseline_accuracy.json',
    'final_accuracy':'results/fp32_compression/head640_head_ce_accuracy.json',
    'baseline_runtime':'results/fp32_compression/runtime/original224_fp32_summary.json',
    'final_runtime':'results/fp32_compression/runtime/head640_head_ce_summary.json',
    'recipe':str((bundle/'recipe.json').as_posix()),'manifest':str((bundle/'manifest.json').as_posix()),
    'technical_report':'output/pdf/best192_fp32_technical_report.pdf'}
  }
 }
 output=ROOT/'metrics.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 # Keep the detailed winner evidence intact; provide a distinct Section 8 copy in the bundle.
 (ROOT/bundle/'submission_metrics.json').write_bytes(output.read_bytes())
 print(json.dumps({'team_name':report['team_name'],'final':final,'worst_class_accuracy':report['worst_class_accuracy'],'codecarbon_mode':report['codecarbon_mode'],'output':'metrics.json'},indent=2))

if __name__=='__main__':main()
