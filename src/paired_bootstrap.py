"""Conditional paired image bootstrap. Fixed fitted models, seeds and thresholds.
No patient-level validation, retraining uncertainty or selection correction.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
SEEDS=(2026,2027,2028)
METRICS=('grade2_sensitivity','referable_sensitivity','specificity','referable_auc')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weighted_auc(scores,y,weights):
 order=np.argsort(scores,kind='stable');v=scores[order];starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
 w=weights[:,order];y=y[order].astype(float)
 pos=np.add.reduceat(w*y,starts,axis=1);neg=np.add.reduceat(w*(1-y),starts,axis=1)
 return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))
def self_test():
 s=np.array([.1,.5,.5,.9]);y=np.array([0,1,0,1]);w=np.array([[1,1,1,1],[2,3,4,1]])
 values=[]
 for row in w:
  num=den=0.
  for i in np.flatnonzero(y):
   for j in np.flatnonzero(1-y):
    num+=row[i]*row[j]*((s[i]>s[j])+.5*(s[i]==s[j]));den+=row[i]*row[j]
  values.append(num/den)
 np.testing.assert_allclose(weighted_auc(s,y,w),values,atol=1e-15)
 assert (np.array([.5,.49])>=.5).tolist()==[True,False]
def read_run(directory,modality,ids=None,grades=None):
 directory=Path(directory);p=directory/f'validation_predictions_{modality}.csv';m=directory/f'validation_metrics_{modality}.json';t=directory/'locked_validation_thresholds.json'
 with p.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 byid={r['sample_id']:r for r in rows}
 if len(byid)!=len(rows):raise ValueError('Duplicate image identifiers')
 ids=tuple(sorted(byid)) if ids is None else ids
 if set(byid)!=set(ids):raise ValueError('Image membership differs')
 rows=[byid[i] for i in ids];g=np.array([int(r['grade_true']) for r in rows])
 if grades is not None and not np.array_equal(g,grades):raise ValueError('Grades differ')
 prob=np.array([[float(r[f'prob_grade_{i}']) for i in range(5)] for r in rows])
 if not np.isfinite(prob).all() or np.any(prob<0):raise ValueError('Invalid probabilities')
 np.testing.assert_allclose(prob.sum(axis=1),1,atol=3e-7,rtol=0)
 score=prob[:,2:].sum(axis=1);meta=json.loads(m.read_text());threshold=json.loads(t.read_text())[modality]['threshold']
 assert threshold==meta['selected_threshold_specificity_0_90']
 decision=(score>=threshold).astype(float)
 point={'grade2_sensitivity':decision[g==2].mean(),'referable_sensitivity':decision[g>=2].mean(),'specificity':(1-decision)[g<2].mean(),
        'referable_auc':weighted_auc(score,g>=2,np.ones((1,len(g)),dtype=int))[0]}
 keys={'grade2_sensitivity':'grade2_sensitivity_at_selected_threshold','referable_sensitivity':'sensitivity_at_specificity_0_90','specificity':'specificity_at_selected_threshold','referable_auc':'referable_auc'}
 for name,key in keys.items():
  if name=='grade2_sensitivity' and modality=='UWF' and key not in meta:continue
  if abs(point[name]-meta[key])>2e-12:raise ValueError(f'Endpoint replay failed {directory} {name}')
 return ids,g,score,decision,point,{'prediction_path':str(p.resolve()),'prediction_sha256':sha(p),'metrics_sha256':sha(m),'thresholds_sha256':sha(t),'threshold':threshold}
def registry(n3,n2=None,component=None,matched=None,design=None):
 models={f'N3/{v}':{s:Path(n3)/f'seed{s}'/v for s in SEEDS} for v in ('M0R','M1_attention','N3_adaptive_s4')}
 comparisons=[('N3','N3/N3_adaptive_s4','N3/M0R'),('N3','N3/N3_adaptive_s4','N3/M1_attention')];missing=[]
 if n2:
  for v in ('M0R','M1_attention','M2_multiscale','N2_local_contrast','A_N2_global_only','A_N2_no_antagonism','A_N2_single_scale'):
   prefix='reused_B1_controls/runs' if v in ('M0R','M1_attention','M2_multiscale') else 'runs'
   models[f'N2/{v}']={s:Path(n2)/prefix/f'seed{s}'/v for s in SEEDS}
   if v!='N2_local_contrast':comparisons.append(('N2','N2/N2_local_contrast',f'N2/{v}'))
 if design:
  names=('M0R','M1_attention','M1_s4','M1_s16','N2_local_contrast','N2_s4','N2_s16','N2_eps_0p0001','N2_eps_0p01','N2_rho_0p05','N2_rho_0p20','N2_sigma_0p75','N2_sigma_1p25')
  for v in names:models['design/'+v]={s:Path(design)/'runs'/f'seed{s}'/v for s in SEEDS}
  for candidate,control in (('N2_local_contrast','M1_attention'),('N2_s4','M1_s4'),('N2_s16','M1_s16')):
   comparisons.extend([('design','design/'+candidate,'design/M0R'),('design','design/'+candidate,'design/'+control)])
  for v in names[7:]:comparisons.append(('design','design/'+v,'design/N2_local_contrast'))
 for root,group,variants in ((component,'component',('M1_generic_s4','N3_fixed_s4','N3_pooled_s4','N3_unsigned_s4')),
                              (matched,'matched',('M1_budget_s4','N3_late_pooled_s4','N3_magnitude_basis_s4'))):
  for v in variants:
   dirs={s:Path(root)/'runs'/f'seed{s}'/v for s in SEEDS} if root else {}
   if root and all((d/'validation_predictions_CFP.csv').is_file() and (d/'validation_predictions_UWF.csv').is_file() for d in dirs.values()):
    models[f'{group}/{v}']=dirs;comparisons.append((group,'N3/N3_adaptive_s4',f'{group}/{v}'))
   else:missing.append({'group':group,'control':v,'status':'LOCAL_PREDICTIONS_NOT_PROVIDED'})
 return models,comparisons,missing
def analyze(n3,output,n2=None,component=None,matched=None,replicates=2000,seed=20261012,design=None):
 self_test();models,comparisons,missing=registry(n3,n2,component,matched,design);rng=np.random.default_rng(seed);sources=[];result=[]
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 for modality in ('CFP','UWF'):
  ids=g=None;loaded={}
  for name,runs in models.items():
   loaded[name]=[]
   for s,d in runs.items():
    ids,g,score,decision,point,source=read_run(d,modality,ids,g);loaded[name].append((score,decision,point));sources.append(dict(source,model=name,seed=s,modality=modality))
  weights=np.zeros((replicates,len(g)),dtype=np.int32)
  for grade in range(5):
   idx=np.flatnonzero(g==grade)
   if not len(idx):raise ValueError('Empty grade stratum')
   weights[:,idx]=rng.multinomial(len(idx),np.ones(len(idx))/len(idx),size=replicates)
  masks={'grade2_sensitivity':g==2,'referable_sensitivity':g>=2,'specificity':g<2};boot={}
  for name,runs in loaded.items():
   boot[name]=[]
   for score,decision,point in runs:
    values={k:weights[:,mask]@((1-decision) if k=='specificity' else decision)[mask]/mask.sum() for k,mask in masks.items()}
    values['referable_auc']=weighted_auc(score,g>=2,weights);boot[name].append(values)
  for family,a,b in comparisons:
   for metric in METRICS:
    delta=np.array([loaded[a][i][2][metric]-loaded[b][i][2][metric] for i in range(3)])
    samples=np.vstack([boot[a][i][metric]-boot[b][i][metric] for i in range(3)])
    unit='AUROC' if metric=='referable_auc' else 'percentage_points';scale=1 if unit=='AUROC' else 100
    scopes=[('fixed_three_seed_mean',delta.mean(),samples.mean(axis=0))]+[(f'seed{s}',delta[i],samples[i]) for i,s in enumerate(SEEDS)]
    for scope,estimate,resampled in scopes:
     low,high=np.quantile(resampled,[.025,.975],method='linear')
     result.append(dict(family=family,candidate=a.split('/')[1],control=b.split('/')[1],modality=modality,metric=metric,scope=scope,effect=float(estimate*scale),ci95_low=float(low*scale),ci95_high=float(high*scale),unit=unit,bootstrap_replicates=replicates,analysis_unit='image',patient_independence_verified=False))
  print(f'{modality}: {len(g)} paired images; {len(comparisons)} contrasts',flush=True)
 with (output/'paired_effects.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
 protocol={'seed':seed,'replicates':replicates,'strata':'five DR grades within modality; counts fixed','pairing':'identical image multiplicities for all fitted models and seeds','estimand':'mean within-seed candidate-minus-control effect; no ensembling',
           'interval':'pointwise percentile 95%, no multiplicity correction','limitations':['Image exchangeability assumed; patient dependence unresolved','No retraining uncertainty or seed resampling','No correction for checkpoint/threshold selection on the same images','No equivalence or biological-failure conclusion from zero crossing'],
           'available_comparisons':len(comparisons),'source_manifest':sources,'missing_comparisons':missing,'script_sha256':sha(__file__)}
 (output/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n3-runs',required=True);p.add_argument('--n2-root');p.add_argument('--component-root');p.add_argument('--matched-root');p.add_argument('--design-root');p.add_argument('--output',required=True);p.add_argument('--replicates',type=int,default=2000);a=p.parse_args()
 if a.replicates<2000:p.error('At least 2000 replicates required')
 analyze(a.n3_runs,a.output,a.n2_root,a.component_root,a.matched_root,a.replicates,design=a.design_root)
