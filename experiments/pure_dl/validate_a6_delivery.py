"""Final delivery structure, integrity and population validator; no model inference."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile
import sys
DELIVERY=Path(__file__).resolve().parents[3]/"submission_a6_heads_fold5_80_501552"
sys.path.insert(0,str(DELIVERY))
from predict import load_inputs


def digest(data):return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(data.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=reject)


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--code_zip',type=Path,required=True)
    ap.add_argument('--result',type=Path,required=True,help='Result JSON, with sibling result.zip, or result.zip with sibling result.json')
    ap.add_argument('--data_dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    checks={}
    with zipfile.ZipFile(args.code_zip) as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        assert len(names)==len(set(names)), 'Duplicate ZIP member'
        for name in names:
            part=PurePosixPath(name)
            assert not part.is_absolute() and '..' not in part.parts and '\\' not in name and ':' not in name
        files={entry.filename for entry in entries if not entry.is_dir()}
        roots={'README.md','predict.py','requirements.txt','runtime_versions.json'}
        assert roots.issubset(files)
        assert all(name in roots or name.startswith(('src/','models/')) for name in files), 'Unexpected outer directory or root file'
        assert 'src/model.py' in files and 'models/MANIFEST.json' in files
        weights={name for name in files if name.endswith('.pt')}
        expected_weights={'models/A.pt','models/B.pt','models/D.pt','models/heads.pt'}
        assert weights==expected_weights,'Must contain A/B/D and independent head weights'
        manifest=strict_json(archive.read('models/MANIFEST.json'))
        assert set(manifest)==files-{'models/MANIFEST.json'},'Manifest must cover every file except itself'
        for name,expected in manifest.items():
            assert digest(archive.read(name))==expected, name
        summary=strict_json(archive.read('models/delivery_summary.json'))
        assert summary['trained_folds']==[5] and not summary['complete_five_folds']
        assert summary['head_best_epochs']==[4,0,11]
        for name,record in summary['source_checkpoints'].items():
            assert digest(archive.read('models/'+name))==record['sha256']
        final=strict_json(archive.read('models/evidence/packaged_fold5_verification.json'))
        assert final['packaged_predictions_exactly_equal_saved'] and final['rows']==144665 and final['wells']==16
        assert final['scores']==summary['validation_fold5_scores']
        checks['code_files']=len(files)
        checks['checkpoints']=sorted(expected_weights)
        checks['manifest_all_hashes_match']=True
        checks['source_checkpoint_hashes_match']=True
        checks['retained_fold5_scores']=summary['validation_fold5_scores']
    if args.result.suffix.lower()=='.zip':
        result_zip=args.result;result_json=args.result.parent/'result.json'
    else:
        assert args.result.suffix.lower()=='.json'
        result_json=args.result;result_zip=args.result.parent/'result.zip'
    raw=result_json.read_bytes()
    with zipfile.ZipFile(result_zip) as archive:
        assert archive.namelist()==['result.json'], 'Result ZIP must contain only root result.json'
        assert archive.read('result.json')==raw,'ZIP/JSON byte mismatch'
    data=strict_json(raw)
    assert type(data) is dict and set(data)=={'resultData'}
    assert type(data['resultData']) is list
    expected=load_inputs(args.data_dir)
    assert len(data['resultData'])==len(expected)
    seen=set();total=0
    bounds={'POR':(0.,50.),'PERM':(.01,10000.),'SW':(8.305,99.9)}
    for item,(well,frame) in zip(data['resultData'],expected):
        assert type(item) is dict and set(item)=={'logId','predictions'}
        assert type(item['logId']) is str and item['logId']==well and well not in seen
        seen.add(well)
        predictions=item['predictions']
        assert type(predictions) is list and len(predictions)==len(frame)
        for index,(row,depth) in enumerate(zip(predictions,frame['DEPTH'].to_numpy())):
            assert type(row) is dict and set(row)=={'DEPTH','POR','PERM','SW'},(well,index)
            assert all(number(value) for value in row.values()),(well,index,'nonnumeric/nonfinite/bool')
            assert math.isfinite(float(depth))
            # Predictor preserves original depth order and serializes six decimals.
            assert row['DEPTH']==round(float(depth),6),(well,index,'depth/order mismatch')
            for name,(low,high) in bounds.items():assert low<=row[name]<=high,(well,index,name)
        total+=len(predictions)
    checks.update({'passed':True,'wells':len(expected),'rows':total,'exact_result_schema':True,
        'source_well_row_depth_order_match':True,'result_zip_json_bytes_match':True,
        'result_sha256':digest(raw),'code_zip_sha256':digest(args.code_zip.read_bytes()),
        'result_zip_sha256':digest(result_zip.read_bytes()),
        'scope':'Final package hashes, supplied saved verification evidence, exact output schema/population/ranges only. Does not execute inference or establish prediction accuracy; fresh extracted CLI execution is evidenced separately by parent logs.',
        'inference_executed_by_validator':False,'test_accuracy':'unknown; test labels unavailable'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(checks,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(checks,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
