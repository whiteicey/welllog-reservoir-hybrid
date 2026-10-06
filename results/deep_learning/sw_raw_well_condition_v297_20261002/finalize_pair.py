"""CPU-only terminal evidence for the bank-v-reference causal contrast."""
import json,runpy
from pathlib import Path
STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[2]
runpy.run_path(str(STUDY/'assemble_snapshot.py'))
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
reports={a:read(ROOT/f'outputs/pure_dl_sw_raw_well_condition_v297_fold5/{a}/report.json') for a in ('well','reference')}
strata=read(STUDY/'STRATA_DIAGNOSTIC.json')
timing=read(STUDY/'BATCH_TIMING.json')
finished=max((ROOT/f'outputs/pure_dl_sw_raw_well_condition_v297_fold5/{a}/report.json').stat().st_mtime for a in reports)
timing.update(terminal=True,total_paired_wall_seconds=finished-min(timing['well_start_epoch'],timing['reference_start_epoch']),individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},measurement_scope='Staggered sequential arms with different overlap against296, includes data/initial/train/finalrestore and launch gap, excludes codeprep/CPUassembly; not isolated speed comparison')
result=dict(reports=reports,strata=strata,well_minus_reference=reports['well']['target_score']-reports['reference']['target_score'],well_minus_reference_continuous=strata['v297_well']['continuous_SW']-strata['v297_reference']['continuous_SW'],gain_vs289=reports['well']['target_score']-74.1537264289993,timing=timing,remaining_folds_started=False,complete_five_folds=False,conclusion='Small matched overall whole-well condition gain; continuous absolute score still below retained289 and252. Preserve overall positive as separate candidate, do not claim continuous breakthrough or sweep bank descriptors/bounds.')
for name,value in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
recipe=read(STUDY/'WELL_EIGHT_STAGE_RECIPE.json')
recipe['matched_well_minus_reference']=result['well_minus_reference']
recipe['matched_continuous_delta']=result['well_minus_reference_continuous']
(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
note=f"\n\n## Final matched contrast\nWell SW{reports['well']['target_score']:.12f}, reference{reports['reference']['target_score']:.12f}; matched well-minus-reference{result['well_minus_reference']:+.12f}. Continuous well{strata['v297_well']['continuous_SW']:.12f}, reference{strata['v297_reference']['continuous_SW']:.12f}, matched{result['well_minus_reference_continuous']:+.12f}; BOTH must be compared to retained28917.72012680313723/25217.88486813991017 before declaring continuous gain. Optional8 actualTOTAL{recipe['scores']['TOTAL']:.12f}, strict808 false; separate positive candidate and matched control retained, no additional stages/folds. Individual well{reports['well']['seconds']:.3f}s/reference{reports['reference']['seconds']:.3f}s, pairwall{timing['total_paired_wall_seconds']:.3f}s incl stagger and overlap against296, not speed ranking. Bank scope evidence applies only to newraw FiLM, not globally to all lithology/fluid modeling. Further descriptor/bound sweeps not justified by this small gain; next priority substantive continuous representation.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k not in ('reports','strata')}))
