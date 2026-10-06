"""Prepare new immutable trial sources and a bounded two-arm protocol."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/perm_adaptive_share_v298_20261002'
STUDY.mkdir(parents=True,exist_ok=True)
for name in ('perm_adaptive_share_v298.py','train_perm_adaptive_share_v298.py','record_perm_adaptive_share_v298.py'):
    if (CODE/name).exists():raise RuntimeError(f'Refuse overwrite: {name}')
source=(CODE/'perm_sparse_reacquisition_v291.py').read_text(encoding='utf-8')
source=source.replace('perm_sparse_reacquisition_v291','perm_adaptive_share_v298').replace("for s in ('point','window')","for s in ('scalar','conditional')")
source=source.replace('super().__init__(master,metadata,mode)','super().__init__(master,metadata,\'point\')\n        self.share_mode=mode')
source=source.replace('self.last_support=0.;self.last_mask_variation=0.', '''self.share=nn.Linear(48,1)
        nn.init.zeros_(self.share.weight);nn.init.zeros_(self.share.bias)
        if mode=='scalar':self.share.weight.requires_grad_(False)
        self.last_support=0.;self.last_mask_variation=0.
        self.last_alpha_min=0.;self.last_alpha_mean=0.;self.last_alpha_max=0.''')
source=source.replace("name='PERM_direct_categorical_readout')]", "name='PERM_direct_categorical_readout'),dict(params=[p for p in self.share.parameters() if p.requires_grad],lr=3e-4,name='PERM_adaptive_share')]")
source=source.replace("return info\n    def forward", "info.update(selection_granularity='point', matched_point_window_initialization=False, primary_control='scalar learned share versus feature conditional share', registered_capacity_same=True, active_capacity_same=False, extra_share_active_parameters=1 if self.share_mode=='scalar' else 49, nominal_alpha_max=.3, actual_alpha_bounds=[0.,.6], bounded_alpha_formula='nominal_ramp*(1+tanh(raw/.3))', old_gate_frozen=True, no291repair_weights_loaded=True)\n        return info\n    def forward")
source=source.replace('mass=((1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)).clamp_min(1e-12)', '''alpha=self.mix_alpha*(1+torch.tanh(self.share(h).float()/.3))
        if self.training:
            actual=alpha[valid].detach()
            self.last_alpha_min=float(actual.min());self.last_alpha_mean=float(actual.mean());self.last_alpha_max=float(actual.max())
        mass=((1-alpha)*prior+alpha*logits.softmax(-1)).clamp_min(1e-12)''')
(CODE/'perm_adaptive_share_v298.py').write_text(source,encoding='utf-8')
train=(CODE/'train_perm_sparse_reacquisition_v291.py').read_text(encoding='utf-8').replace('perm_sparse_reacquisition_v291','perm_adaptive_share_v298')
train=train.replace("training_sparse_support=model.last_support,", "training_actual_alpha_min=model.last_alpha_min,training_actual_alpha_mean=model.last_alpha_mean,training_actual_alpha_max=model.last_alpha_max,training_sparse_support=model.last_support,")
train=train.replace("'perm_direct_share_v283.py'", "'perm_sparse_reacquisition_v291.py','perm_direct_share_v283.py'")
(CODE/'train_perm_adaptive_share_v298.py').write_text(train,encoding='utf-8')
record=(CODE/'record_perm_sparse_reacquisition_v291.py').read_text(encoding='utf-8').replace('perm_sparse_reacquisition_v291','perm_adaptive_share_v298').replace('V291','V298')
(CODE/'record_perm_adaptive_share_v298.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/perm_sparse_reacquisition_v291_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['scalar','conditional'],primary_control='same registered49param share module, 1 active scalar versus49 conditional; body/readout initialization exactsame',reference_best=89.5155547932572,extra_active_parameters={'scalar':1,'conditional':49},trainable_parameters={'scalar':36467,'conditional':36515},alpha_bounds=[0.,.6],nominal_alpha_ramp_max=.3,only_original154_weights=True,no291trained_weights=True,prepared_not_launched=True)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original154 frozen prior only; same new291 sparse architecture/init, allnew encoder/readout/share jointly train. Scalar versus conditional share only; 48 activeparameter difference disclosed. Max2GPU/80epochs/patience14/min10/originalQ50/split/scoring. One addedPERMstage/max8, no291trained weights, no otherfolds before strict80.8 and usefulgain. Maintain PROJECTS, independentREADME/EXPLORATIONS/atlas. No frozen source or delivered files changed.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# PERM bounded adaptive component share v298

## Focus / predefined hypothesis
The positive291 sparse raw encoder jointlylearns a direct continuous density but mixes it at fixed .3 even where it is less/more reliable than original154 prior. Learn bounded component share jointly with this fresh encoder/readout: alpha=nominal_ramp*(1+tanh(raw/.3)), nominal0->.3 over5epochs, actual[0,.6]. Original154 gate/prior frozen, no291repair weights. No architecture/backbone replacement of protected prior, no teacher or validation-guided perrow selection.

Scalar control learns one bias (48 zero share weights frozen); conditional arm learns the same zero-init48->1 module. Both same registered parameters and original291 random initialization/inference/protocol, active36467 versus36515: disclose48parameter difference, not exact active-capacity ablation. Allnew sparse body/readout plus active share parameters jointly3e-4. DirectcomponentCE1 remains independent ofalpha to prevent branch starvation; mixedNLL/equal-trainingwellrisk/jointRPS1 learns reliability through training labels only. Both arms pointselector only, oldPERM default gate frozen, hence objective targets continuous density. Same original206+relative9curves/Q50/originalfold5/seed20261010/80-14/min10/b64/64steps.

## Explored and verification
Prepared, untrained. Ordinary initial evaluation must replay154 within1e-3, actual newgroup gradients recorded; no additional preflight. Atomic laststate includes actual learnedshare weights, nominal schedule buffer, optimizers/RNG and own-best state. Nominalalpha .3 is NOT actual learnedalpha; training min/mean/max logged from actualvalid inputs. OfficialPERM-best selection and actual144665 assembly after completion, RAM replay is not independent diskreload.

## Gain
Unproved. Primary conditional-minus-scalar finalPERM; absolute retained29189.5155547932572 and default15489.40556485691894. One addedPERMphase from original154, not chained onto trained291; whole method<=8. No strict80.8 pass/fullOOF82 claim, no remainingfolds. Earlier adaptive density/background routers existed; this is new directcomponent reliability migration, not first adaptive mixture.

## Not worth continuing
If conditional fails scalar/retained and continuous metric does not improve, close this bounded share setting, no boundary/temperature/LR grid. Shortcut feature-dependent allocation can fit trainwells; broad conditional densities may deserve averaging instead of pointwise confidence. A useful scalar gain alone does not prove conditional mechanism.

## Timing
Record individual data/initial/train/finalRAMreplay seconds/peakmemory and same-work paired wall including staggered overlap/gaps; exclude codeprep and CPU assembly. Max2 is a limit, not automatically a speedup.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: useful continuous PERM gain within one newstage. Done: twoarm bounded protocol and immutable fresh sources. Pending: explicit launch scalar/conditional, terminal ownbest assembly/paired diagnostics/timing, retention and atlas. After useful strict80.8 combination only: qualify and remaining originalfolds toward OOF>82.\n',encoding='utf-8')
print(json.dumps({'study':str(STUDY),'sources':{n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in ('perm_adaptive_share_v298.py','train_perm_adaptive_share_v298.py')}}))
