"""Configurable public adapter; not the immutable private server replay."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'vendor/stage25e'))
def main(a):
 import torch
 from torchvision.models import efficientnet_b0
 import train_stage25e_cfp_grade2_controlled as baseline
 from bio_vision_model import BioVisionModel
 from training_functions import make_loader,train_epoch,evaluate
 from design_space import CONFIGS
 if a.variant not in CONFIGS:raise ValueError('Unknown variant')
 if a.output.exists():raise RuntimeError('Use a new output directory')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one CUDA GPU')
 spec=json.loads((R/'configs/training_defaults.json').read_text());frames={m:{} for m in ('CFP','UWF')}
 for m in frames:
  for split in ('train','validation'):
   frame=pd.read_csv(a.data_dir/f'{m.lower()}_{split}.csv',dtype={'sample_id':str})
   required=['sample_id','image_path','grade']+[f'lesion_{n}{s}' for n in baseline.LESIONS for s in ('','_mask')]
   if set(required)-set(frame) or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():raise ValueError('Manifest schema/IDs invalid')
   if not frame.grade.isin(range(5)).all() or not all(Path(p).is_file() for p in frame.image_path):raise ValueError('Grade/image path invalid')
   for n in baseline.LESIONS:
    mask=frame[f'lesion_{n}_mask']
    if not mask.isin((0,1)).all() or not frame.loc[mask==1,f'lesion_{n}'].isin((0,1)).all():raise ValueError('Lesion labels/masks invalid')
   frames[m][split]=frame
  if set(frames[m]['train'].sample_id)&set(frames[m]['validation'].sample_id) or set(frames[m]['train'].image_path)&set(frames[m]['validation'].image_path):raise ValueError('Image overlap')
 digest=hashlib.sha256(a.imagenet_weights.read_bytes()).hexdigest()
 if digest!=json.loads((R/'configs/weight_source.json').read_text())['sha256']:raise RuntimeError('ImageNet weight hash changed')
 imagenet=torch.load(a.imagenet_weights,map_location='cpu')
 def offline_encoder(*,weights=None):
  net=efficientnet_b0(weights=None);net.load_state_dict(imagenet,strict=True);return net
 baseline.efficientnet_b0=offline_encoder;baseline.set_seed(a.seed)
 legacy=baseline.Stage20CModel('matched_baseline',256,160);common={k:v.detach().cpu().clone() for k,v in legacy.state_dict().items()};del legacy
 baseline.set_seed(a.seed);model=BioVisionModel(a.variant,imagenet);missing,unexpected=model.load_state_dict(common,strict=False)
 if unexpected or any('.stage_attention.' not in k for k in missing):raise RuntimeError('Common initialization mismatch')
 model.use_ordinal=False;device=torch.device('cuda:0');model.to(device);encoder=[];heads=[]
 for n,v in model.named_parameters():(encoder if '.encoder.' in n else heads).append(v)
 opt=torch.optim.AdamW([{'params':encoder,'lr':spec['encoder_lr']},{'params':heads,'lr':spec['head_and_attention_lr']}],weight_decay=spec['weight_decay'])
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=spec['scheduler_T_max'])
 criteria={k+'_'+m:fn(frames[m]['train']).to(device) for m in frames for k,fn in [('grade',baseline.grade_weights),('lesion',baseline.lesion_pos_weights)]}
 loaders={m:make_loader(torch,baseline,frames[m]['validation'],spec,False,20260926+i*10+1) for i,m in enumerate(('CFP','UWF'))}
 a.output.mkdir(parents=True);history=[];streams=[];best_key=best_state=None;best_epoch=stale=0
 for epoch in range(1,spec['epochs_max']+1):
  start=time.time();loss,audit=train_epoch(torch,baseline,model,frames,opt,criteria,device,spec,a.seed,epoch)
  metrics={m:evaluate(torch,baseline,model,m,loaders[m],frames[m]['validation'],device,spec,a.seed)[0] for m in frames};key=baseline.cfp_checkpoint_selection_key(metrics['CFP'])
  if baseline.is_better_cfp_selection(key,best_key):best_key=key;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:stale+=1
  history.append(dict(epoch=epoch,seconds=time.time()-start,cfp_grade2=key[0],cfp_auc=key[2],**loss));streams.append(dict(epoch=epoch,**audit));sched.step()
  print(f'epoch={epoch} best={best_epoch} CFP_grade2={key[0]:.6f}',flush=True)
  if stale>=spec['early_stopping_patience']:break
 if best_state is None:raise RuntimeError('No selected checkpoint')
 model.load_state_dict(best_state,strict=True);torch.save({'model_state':best_state,'variant':a.variant,'seed':a.seed,'best_epoch':best_epoch,'pretrained_sha256':digest},a.output/'best_model.pt')
 thresholds={}
 def convert(v):
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,np.ndarray):return v.tolist()
  raise TypeError(type(v).__name__)
 for m in frames:
  metrics,pred=evaluate(torch,baseline,model,m,loaders[m],frames[m]['validation'],device,spec,a.seed)
  pred['variant']=a.variant;pred.to_csv(a.output/f'validation_predictions_{m}.csv',index=False,float_format='%.17g')
  (a.output/f'validation_metrics_{m}.json').write_text(json.dumps(metrics,indent=2,default=convert)+'\n')
  thresholds[m]={'threshold':metrics['selected_threshold_specificity_0_90']}
 (a.output/'locked_validation_thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
 pd.DataFrame(history).to_csv(a.output/'training_history.csv',index=False,float_format='%.17g')
 (a.output/'training_stream_audit.json').write_text(json.dumps(streams,indent=2)+'\n')
 (a.output/'adapter_status.json').write_text(json.dumps({'private_replay_verified':False,'patient_independence_verified':False,'external_or_test_used':False,'best_epoch':best_epoch},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,choices=[2026,2027,2028],required=True)
 p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--imagenet-weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
