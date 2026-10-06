from pathlib import Path
import json,numpy as np,torch
S=Path(__file__).resolve().parent;R=S.parents[2];torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
d=torch.load(R/"results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt",map_location="cpu",weights_only=False)
y=np.asarray(d["truth"][d["validation"]]);p=np.load(S/"eight_stage_pred.npy");old=read(R/"results/deep_learning/perm_ranked_probability_v154_20261001/CURRENT_RECIPE.json")
base=np.column_stack([np.load((R/old[t]["checkpoint"]).parent/"channel_pred.npy") for t in ("POR","PERM","SW")])
def utilities(a):
 return np.c_[np.maximum(0,1-abs(a[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),np.maximum(0,1-abs(np.log10(np.maximum(a[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),np.maximum(0,1-abs(a[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]
u=utilities(p);ub=utilities(base);mask=np.c_[y[:,0]<=.1,y[:,1]<=.01,y[:,2]>=99.9];recipe=read(S/"CURRENT_EIGHT_STAGE_RECIPE.json");gap=80.8-recipe["scores"]["TOTAL"];result={"screening_fold":5,"diagnostic_only":True,"no_oracle_selection_or_training_reweighting":True,"remaining_folds_started":False,"strict_gate_gap":gap,"channels":{}}
lines=["# Current eight-stage continuous-value bottleneck diagnosis","","Fixed original fifth, 144665 rows. Validation-label analysis only; it must not define per-row models, weights or train features. No new training phase, no oracle selections, no test labels. Results describe this selected recipe and do not prove backbone causality or generalization.","","|Target|Continuous rows|Continuous score|Continuous zero utility|Default score|Whole gain vs default5|","|---|---:|---:|---:|---:|---:|"]
for c,(t,w) in enumerate(zip(("POR","PERM","SW"),(.3,.35,.35))):
 strata={}
 for name,sel in (("default",mask[:,c]),("continuous",~mask[:,c])):
  v=u[sel,c];delta=u[sel,c]-ub[sel,c]
  strata[name]={"rows":int(sel.sum()),"fraction":float(sel.mean()),"score":float(v.mean()*100),"baseline_score":float(ub[sel,c].mean()*100),"gain":float(delta.mean()*100),"zero_utility_fraction":float((v==0).mean()),"partial_utility_fraction":float(((v>0)&(v<1)).mean()),"weighted_total_gain_contribution":float(delta.sum()/len(y)*100*w)}
 result["channels"][t]={"weight":w,"strata":strata,"whole_score":float(u[:,c].mean()*100),"whole_gain":float((u[:,c]-ub[:,c]).mean()*100),"one_channel_whole_gain_to_gate":gap/w,"CONT_only_gain_to_gate_with_default_fixed":gap/(w*strata["continuous"]["fraction"])}
 a=result["channels"][t];q=strata["continuous"];lines.append(f"|{t}|{q['rows']}|{q['score']:.6f}|{q['zero_utility_fraction']:.2%}|{strata['default']['score']:.6f}|{a['whole_gain']:+.6f}|")
 lines.extend([])
lines.extend(["",f"Current TOTAL {recipe['scores']['TOTAL']:.12f}; reaching80.8 needs {gap:.12f} more TOTAL and the gate requires strictly above it.","","With all other scores fixed, the required continuous-stratum gain for each target is:"])
for t,a in result["channels"].items():lines.append(f"- {t}: +{a['CONT_only_gain_to_gate_with_default_fixed']:.6f} CONT points (whole-target +{a['one_channel_whole_gain_to_gate']:.6f}). Arithmetic opportunity only, not a feasible or promised gain.")
lines.extend(["","The low continuous score alone cannot establish a weak backbone: original official tolerance, default/continuous population, ambiguity across wells, representation and likelihood-to-decision readout all contribute. New hypotheses must change encoder and readout jointly and be tested from original allowed sources, never chain retained trained repairs. Preserve DEFAULT performance when measuring CONT improvements. No new architecture is selected by this diagnostic; no row-wise or per-well oracle predictions created.","","Completed: baseline/retained whole and strata comparison. Not done: held-out full five-fold proof, causal backbone-versus-readout attribution, further training candidate. Closure: do not repeatedly recompute unchanged diagnostics or treat validation near-miss distribution as train labels."])
(S/"BOTTLENECK_DIAGNOSTIC.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(S/"BOTTLENECK_DIAGNOSTIC.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(result))
