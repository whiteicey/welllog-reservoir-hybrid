"""Prepare fresh scoped interaction pair; preserve all historical attempts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/por_curve_scope_v302_20261002';STUDY.mkdir(parents=True,exist_ok=True)
path=CODE/'train_por_curve_scope_v302.py'
if path.exists():raise RuntimeError('Refuse overwrite')
train=(CODE/'train_por_direct_share_v290.py').read_text(encoding='utf-8').replace('por_direct_share_v290','por_curve_scope_v302')
train=train.replace("'por_direct_encoder_v287.py'","'por_direct_share_v290.py','por_direct_encoder_v287.py'")
path.write_text(train,encoding='utf-8')
record=(CODE/'record_por_direct_share_v290.py').read_text(encoding='utf-8').replace('por_direct_share_v290','por_curve_scope_v302').replace('V290','V302')
(CODE/'record_por_curve_scope_v302.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/por_direct_share_v290_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['grouped','full'],primary_control='same numeric9token module/tensors/capacity; 3groups of3 attention versusall9',trainable_parameters=76592,new_interaction_parameters=19024,registered_and_active_capacity_same=True,no290trained_weights=True,only_original151_checkpoint=True,prepared_not_launched=True)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original151 frozenprior only, fresh290rawbody/readout+new19024interaction jointlytrain, no oldrepairweights. Grouped3x3/full9token sameparams/init, no diagonalonly zero-QK control. 9curves206+relative/Q50/originalfold5/seed/80-14/min10/max2GPU, onePORaddedphase/max8. Maintain README/PROJECTS/EXPLORATIONS/atlas; no frozenfiles/deliveries/otherfolds changed.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# POR curve-token interaction scope v302

## Focus / historical evidence
Curve-token attention already exists in paired_curve_attention_residual_sequence, paired_sw_curve_attention_sequence, curve_token_network andv135. This is NOT first numericcurve token attention. New question: same additional jointencoder module, restricted three-curve groups versus allnine connectivity, in the current protected151+fresh290directdensity architecture. No old attention/290weights; original151 checkpoint only. Research source/oldresults linked when scoped review completes.

Nine original normalizedrawcurves scalar->32 sharednumericembedding +9 learnedidentity vectors; two preLN fourhead32attention/FF64 blocks, meanpool->48zero-output bridge added to new207->48stem before threepointMLPs/fuse/readout. Newmodule computed once perforward, shared by existingbranches; allnew rawbody/readout/19024interaction parameters jointly3e-4, 76592total active. Same existing206derivedfeatures/relative bypass, no additionalcurves/inputlabels. Oldpredictor completely frozen. Same mixing.25ramp5/directcomponentphysical-toleranceCE1/jointRPS1/oldPORgate/80-14/256window. Wholemethod max8, onePORaddedphase from151, not trained290->newstage.

Predefine groups [GR,PE,SP], [AC,DEN,CNL], [CAL,RXO,RT]. Third group mixes holegeometry/resistivity and is an arbitrary input-defined restriction, not asserted true geology. Grouped3tokens keeps Q/K/V effective; self-onlydiagonal rejected because softmax1/QK gradients0 would destroy effectivecapacity matching. Both registered/activeparameter counts/init tensors same, attentionmask onlydifference; bridgezero initiallysourcefunction unchanged. Dense206path and postpool readout still permit cross-feature mixing in BOTH, so this tests explicit token communication scope, not all possible feature interaction versus none.

## Explored and validation
Prepared/not launched. Ordinaryinitial151 replay, actualnewencoder/readout/token group gradients, ownWHOLEPOR-best/frozenproof/atomic current-best RNG; actual144665 composition/strata afterterminal, RAMreplay not independentdiskreload. Historical290 capacity57568 differs, absolute reference only, not samecapacity architectureablation. Input-only grouping, no label fitting or heldoutneighbor retrieval.

## Gain
Unproved. Primary full-minus-grouped finalPOR, absolute29078.0235265552551; meaningfullocalgain can be migrated to a freshonephase candidate, but no automatic stage stacking/otherfolds. Currentoptional8/default5/oldOOFdelivery preserved. StrictTOTAL>80.8 usefulgain before originalremainingfolds, OOF>82 unmet.

## Unworthy followup
If neither group norfull beats retained and no meaningful matchedbenefit, close this communication-scope setting, no group assignment/headcount/width grid. Attentionmay only add capacity or emphasize imputations; original206features already mixcurves. Do not generalize a failure to all FTTransformer/numericfeature attention. Depthattention and fullreleased pretrained models are outside this study.

## Timing
Individual data/initial/train/finalrestore +peakmemory; pairwall includesactualoverlap/stagger/gap, codeprep/CPUassemblyexcluded. Max2GPU/notdefaultspeedup, no redundant preflight/review.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: meaningful explicit curvecommunication benefit via newjointencoder/readout, sameeffectivecapacity pair. Done: historycheck and groupedcontrol avoidsdiagonalzero-QK issue; protocol/source prepared. Pending: explicit launches, finalassembly/pairedstrata/timing, source/evidence/atlas. After usefulstrict808 only: verifyremainingfolds towardfullOOF82.\n',encoding='utf-8')
print(str(STUDY))
