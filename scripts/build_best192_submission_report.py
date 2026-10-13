"""Build the two-page submission report for the selected smaller192 FP32 model."""
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
TMP=ROOT/'tmp/pdfs'
BUNDLE=ROOT/'best_models/resize192_head640_head_ce_fp32_8655a532a076'
REPO='https://github.com/thesid42/minimalist-edge-ai-challenge'

def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
 OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
 selected=read('best_models/current.json')
 assert selected['model_sha256']=='8655a532a07639d47592b634e9596dd36c808ee1632a835fab1f6af8be4612df'
 rows={r['name']:r for r in read('results/fp32_compression/summary.json')['records']}
 runtime={r['name']:r for r in read('results/fp32_compression/runtime/summary.json')['records']}
 score=read('results/fp32_compression/head640_head_ce_accuracy.json')
 recipe=read(str((BUNDLE/'recipe.json').relative_to(ROOT)))
 baseline=rows['original224_fp32'];parent=rows['parent192_fp32'];final=rows['head640_head_ce']
 btime=runtime['original224_fp32'];ftime=runtime['head640_head_ce']
 latency_saving=(1-ftime['median_ms']/btime['median_ms'])*100
 energy_saving=(1-ftime['energy_j_per_inference']/btime['energy_j_per_inference'])*100
 size_saving=(1-final['size_mib']/baseline['size_mib'])*100
 param_saving=(1-final['parameters']/baseline['parameters'])*100
 mac_saving=(1-final['macs']/baseline['macs'])*100
 page1_intro=(f"We select a physically smaller MobileNetV2 with internal 192-pixel inference and FP32 weights. "
  f"It keeps {final['accuracy_pct']:.4f}% accuracy on all 27,000 EuroSAT images while reducing file size "
  f"{size_saving:.1f}%, CPU latency {latency_saving:.1f}% and estimated energy {energy_saving:.1f}% against our unchanged FP32 baseline.")
 method=("Start with the supplied 10-class MobileNetV2 and the saved recovered192 FP32 weights. "
  "The parent was adapted by one classifier-only cross-entropy epoch on all 27,000 images. "
  "Keep the external FP32 input (1,3,224,224); resize internally to 192x192 with bilinear interpolation "
  "(align_corners=False, no antialias). The output stays (1,10), and export uses ONNX opset 13.")
 pruning=("Physically remove 640 of the final 1,280 feature channels. Rank channels by absolute BN scale "
  "times the L2 norm of their classifier-weight column. Torch-Pruning propagates removal through the final "
  "convolution, BN and classifier inputs; earlier layers and residual widths are retained. This removes "
  "212,480 parameters; it does not store zero masks or use reduced-precision weights.")
 training=("Recover the pruned classifier only: 6,410 trainable parameters, one cross-entropy epoch, "
  "learning rate 1e-4, batch 32, seed 42, all 27,000 images. Freeze the backbone and BN running statistics. "
  "Final recovery ran on the RTX 3070 Laptop GPU in FP32, with AMP and TF32 disabled, and took 37.06 seconds. "
  "Export and every reported local inference run use CPU FP32. Only classifier weight and bias changed during recovery.")
 judgement=("Keep classifier-only recovery: pruning starts at 98.9963% accuracy, and recovery restores it to "
  "99.1481% with 97.40% worst-class recall. Discard the separate full-backbone recovery of this same 640-channel "
  "architecture: one epoch at learning rate 5e-5 reduced accuracy to 98.8815% (worst class 97.65%). "
  "No quantization is applied because this selected submission uses FP32 throughout.")
 protocol=("Organizer clarification: train and evaluate on all 27,000 images with our own laptop baseline; "
  "these are training-set accuracies. RGB images are resized to 224 and ImageNet-normalized "
  "(mean 0.485/0.456/0.406; std 0.229/0.224/0.225). Hardware: Ryzen 7 5800H, 27.86 GiB RAM, Windows 10; "
  "ONNX Runtime 1.31.0 CPUExecutionProvider, four threads, batch one. Three serial fresh processes per model "
  "use rotating order, 100 warmups and 1,000 timings of one real image. Accuracy separately covers the entire pool.")
 energy=(f"CodeCarbon 3.3.1 uses CPU TDP/load estimation (45 W) plus a RAM heuristic; GPU is excluded and no training runs concurrently. "
  f"Three 30-second windows give baseline {btime['energy_j_per_inference']:.6f} +/- {btime['energy_stdev']:.6f} J/image "
  f"and final {ftime['energy_j_per_inference']:.6f} +/- {ftime['energy_stdev']:.6f} J/image (mean +/- sample SD). "
  "These are same-laptop estimates, not physical meter readings or edge-device energy. The selected model has no AI Hub profile yet.")
 interpretation=(f"The final model loses only 0.0148 percentage points versus baseline (four images), while worst-class recall "
  f"improves 1.56 points. It is 9.5% smaller and gets eleven more images correct than unpruned recovered192. "
  f"Timing and energy are similar to that parent within run variability; feature pruning does not demonstrate extra CPU acceleration. "
  f"Sampled peak RSS is essentially unchanged versus baseline, so we claim no RAM reduction. THOP MACs exclude resize overhead.")

 comparison=[['Metric','Original 224 FP32','Recovered 192 FP32','Selected smaller 192']]
 specifications=[('Accuracy / 27,000',lambda r,t:f"{r['accuracy_pct']:.4f}%"),
 ('Worst class recall',lambda r,t:f"{r['worst_recall_pct']:.2f}%"),
 ('Parameters',lambda r,t:f"{r['parameters']:,}"),
 ('ONNX size (MiB)',lambda r,t:f"{r['size_mib']:.3f}"),
 ('MACs (million)',lambda r,t:f"{r['macs']/1e6:.3f}"),
 ('CPU median (ms/image)',lambda r,t:f"{t['median_ms']:.4f}"),
 ('Sampled peak RSS (MiB)',lambda r,t:f"{t['peak_rss_mib']:.2f}"),
 ('Estimated energy (J/image)',lambda r,t:f"{t['energy_j_per_inference']:.6f}")]
 for label,fn in specifications:
  comparison.append([label,fn(baseline,btime),fn(parent,runtime['parent192_fp32']),fn(final,ftime)])
 fig,ax=plt.subplots(figsize=(7.05,2.3))
 stages=[('original224_fp32','Original 224',(-10,7),'#647684','o'),
  ('parent192_fp32','Recovered 192',(-10,-14),'#647684','o'),
  ('initial_head640','Pruned, before recovery',(8,6),'#be883e','o'),
  ('head640_ce','Full-backbone recovery: discard',(8,-5),'#b54f50','x'),
  ('head640_head_ce','Classifier recovery: selected',(8,6),'#127d6b','*')]
 for key,label,offset,color,marker in stages:
  r=rows[key];ax.scatter(r['size_mib'],r['accuracy_pct'],color=color,marker=marker,s=95 if marker=='*' else 38,zorder=3)
  ax.annotate(label,(r['size_mib'],r['accuracy_pct']),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if offset[0]<0 else 'left')
 kept=[baseline,parent,rows['initial_head640'],final]
 ax.plot([r['size_mib'] for r in kept],[r['accuracy_pct'] for r in kept],color='#9db9b3',lw=1,ls='--',zorder=1)
 ax.set(xlim=(7.60,8.59),ylim=(98.82,99.22),xlabel='FP32 ONNX size (MiB)',ylabel='Training-set accuracy (%)')
 ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
 ax.grid(alpha=.18);fig.tight_layout(pad=.8)
 curve=TMP/'best192_submission_tradeoff.png';fig.savefig(curve,dpi=210);plt.close(fig)

 fonts=Path('C:/Windows/Fonts')
 regular=fonts/'segoeui.ttf';bold=fonts/'segoeuib.ttf'
 if not regular.exists() or not bold.exists():
  fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
  regular=fonts/'DejaVuSans.ttf';bold=fonts/'DejaVuSans-Bold.ttf'
 pdfmetrics.registerFont(TTFont('Report',regular.as_posix()))
 pdfmetrics.registerFont(TTFont('ReportBold',bold.as_posix()))
 pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold')
 style=getSampleStyleSheet()
 style.add(ParagraphStyle('BodyReport',fontName='Report',fontSize=9.1,leading=12.5,spaceAfter=7,textColor=colors.HexColor('#253746')))
 style.add(ParagraphStyle('SectionReport',fontName='ReportBold',fontSize=11.5,leading=14,spaceBefore=7,spaceAfter=5,textColor=colors.HexColor('#127d6b')))
 style.add(ParagraphStyle('TitleReport',fontName='ReportBold',fontSize=21,leading=25,spaceAfter=5,textColor=colors.HexColor('#203c48')))
 style.add(ParagraphStyle('SubtitleReport',fontName='Report',fontSize=10,leading=13,spaceAfter=11,textColor=colors.HexColor('#617581')))
 style.add(ParagraphStyle('TinyReport',fontName='Report',fontSize=7.5,leading=10,spaceAfter=5,textColor=colors.HexColor('#425662')))
 style.add(ParagraphStyle('CellReport',fontName='Report',fontSize=8.0,leading=10))
 style.add(ParagraphStyle('CardReport',fontName='ReportBold',fontSize=15,leading=18,alignment=TA_CENTER,textColor=colors.HexColor('#127d6b')))
 style.add(ParagraphStyle('CardLabelReport',fontName='Report',fontSize=7.7,leading=10,alignment=TA_CENTER,textColor=colors.HexColor('#425662')))
 def para(text,kind='BodyReport'): return Paragraph(escape(text),style[kind])
 def section(title,text):return [para(title,'SectionReport'),para(text)]
 story=[para('Smaller 192 FP32 MobileNetV2','TitleReport'),para('Technical report | The Minimalist Edge AI Challenge','SubtitleReport'),para(page1_intro)]
 cards=[]
 for value,label in [(f"{final['accuracy_pct']:.3f}%",'all-data accuracy'),(f"{final['size_mib']:.3f} MiB",'FP32 model'),(f"{ftime['median_ms']:.3f} ms",'CPU latency / image'),(f"{ftime['energy_j_per_inference']:.5f} J",'estimated energy / image')]:
  cards.append([Paragraph(value,style['CardReport']),Paragraph(label,style['CardLabelReport'])])
 table=Table([cards],colWidths=[127.7]*4)
 table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf6f2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
 story += [table,Spacer(1,6)]
 story+=section('1. Model and input contract',method)
 story+=section('2. Structural compression and recovery',pruning)
 story.append(para(training))
 story.append(para('3. Matched baseline and final results','SectionReport'))
 cells=[[para(str(value),'CellReport') for value in row] for row in comparison]
 table=Table(cells,colWidths=[144,122,122,123],repeatRows=1)
 table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef1')),('BACKGROUND',(3,1),(3,-1),colors.HexColor('#edf6f2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#98afb6')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe4e8'))]))
 story += [table,Spacer(1,7),para(interpretation)]
 story.append(PageBreak())
 story.append(para('4. Tradeoff curve and selection','SectionReport'))
 story.append(Image(str(curve),width=511,height=166.7))
 story.append(para(judgement))
 story+=section('5. Measurement protocol and energy mode',protocol)
 story.append(para(energy))
 story.append(para('6. Selected model: recall across all ten classes','SectionReport'))
 items=list(score['per_class_recall'].items())
 recall=[['Class','Recall','Class','Recall']]
 for left,right in zip(items[:5],items[5:]):recall.append([left[0],f'{left[1]*100:.2f}%',right[0],f'{right[1]*100:.2f}%'])
 table=Table([[para(str(v),'CellReport') for v in row] for row in recall],colWidths=[183,72,183,73])
 table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef1')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#dbe4e8'))]))
 story.append(table)
 story.append(para('7. Reproduction and delivered evidence','SectionReport'))
 story.append(para('The saved bundle contains model.onnx, the full FP32 recovered_state.pt, exact channel indices and training settings in recipe.json, hash-matched metrics and validation. Restore it using edgeprep.challenge.restore_recipe with the supplied student checkpoint under Models/. To rebuild the pipeline, run scripts/fp32_structural_screen.py build, recover head640 with the classifier-only settings above, export on CPU and evaluate with --limit 0. Exact commands are in docs/fp32_model_reduction.md; replay is checked by tests/test_structural.py.','TinyReport'))
 story.append(Paragraph(f'<link href="{REPO}">{REPO}</link>',style['TinyReport']))
 story.append(para('Selected ONNX SHA-256: '+selected['model_sha256'],'TinyReport'))
 story.append(para('Baseline/final evidence: results/fp32_compression/summary.json and runtime/summary.json. Full-class scores: head640_head_ce_accuracy.json. Source code, weights and all recorded results are in the repository.','TinyReport'))
 def footer(canvas,doc):
  canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d6e1e5'));canvas.line(42,38,553,38)
  canvas.setFont('Report',7.5);canvas.setFillColor(colors.HexColor('#617581'))
  canvas.drawString(42,24,'Minimalist Edge AI | Selected smaller192 FP32 | 13 Oct 2026')
  canvas.drawRightString(553,24,f'{doc.page} / 2');canvas.restoreState()
 target=OUT/'best192_fp32_technical_report.pdf'
 document=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=37,bottomMargin=48,title='Smaller 192 FP32 MobileNetV2 - Technical Report',author='Minimalist Edge AI project')
 document.build(story,onFirstPage=footer,onLaterPages=footer)
 md=['# Smaller 192 FP32 MobileNetV2 - submission technical report','',page1_intro,'','## Model and input contract','',method,'','## Structural compression and recovery','',pruning,'',training,'','## Matched results','','| '+' | '.join(comparison[0])+' |','|---|---:|---:|---:|']
 md += ['| '+' | '.join(row)+' |' for row in comparison[1:]]
 md += ['',interpretation,'','## Tradeoff curve and selection','',judgement,'','The two-page PDF includes the selected pipeline tradeoff curve, including the discarded full-backbone recovery of the same architecture.','',f'[Two-page PDF](../output/pdf/{target.name})','','## Measurement protocol','',protocol,'',energy,'','## Per-class recall','','| Class | Recall |','|---|---:|']
 md += [f'| {name} | {value*100:.2f}% |' for name,value in items]
 md += ['','## Reproduction','',f'[Saved weights and evidence](../{selected["bundle"]}/README.md). [Exact implementation and commands](fp32_model_reduction.md).',f'ONNX SHA-256: `{selected["model_sha256"]}`.','',f'[Repository]({REPO}). Source: challenge PDF sections 2, 3, 5, 6 and 9; the organizer all-data/laptop-baseline clarification governs evaluation.','']
 (ROOT/'docs/submission_best192.md').write_text('\n'.join(md),encoding='utf-8')
 print('Built',target)
 print('Ratios:',json.dumps({'size_reduction_pct':size_saving,'parameter_reduction_pct':param_saving,'mac_reduction_pct':mac_saving,'latency_reduction_pct':latency_saving,'estimated_energy_reduction_pct':energy_saving}))

if __name__=='__main__':main()
