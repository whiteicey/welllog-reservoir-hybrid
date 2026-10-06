"""Create a provenance-recorded trainer without altering completed experiments."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/crosscurve_features_v32_20260929'


def main():
    src=Path(__file__).with_name('train_mechanism_v31.py');dst=Path(__file__).with_name('train_crosscurve_v32.py')
    if dst.exists():raise FileExistsError(dst)
    text=src.read_text(encoding='utf-8-sig')
    edits=[
        ('Two finite mechanism checks on the retained v30; independent candidate states.','Matched redundant/cross-curve feature corrections on retained v31.'),
        ("choices=('POR','SW')","choices=('POR','PERM','SW')"),
        ("module_name='por_reparam_v31' if target=='POR' else 'sw_warp_v31'","module_name='crosscurve_v32'"),
        ("('champion_coadapt_v30.py','champion_split_route_v29.py'","('prepare_crosscurve_v32.py','por_reparam_v31.py','champion_coadapt_v30.py','champion_split_route_v29.py'"),
        ('feature_count=206,amplitude_side=63,train_points=578891','feature_count=206,amplitude_side=63,extra_view_features=18,cached_views=2,train_points=578891'),
        ('v31_candidate_state_plus_documented_frozen_dependencies','v32_feature_head_plus_frozen_v31_dependencies')]
    for old,new in edits:
        if text.count(old)!=1:raise ValueError('Expected exactly one substitution: '+old)
        text=text.replace(old,new,1)
    ast.parse(text);dst.write_text(text,encoding='utf-8')
    report=dict(parent=str(src.relative_to(ROOT)),parent_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        created=str(dst.relative_to(ROOT)),created_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
        changes=edits,historical_source_modified=False)
    (STUDY/'trainer_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
