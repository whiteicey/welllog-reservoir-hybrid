"""Package this user-authorized fifth-fold delivery; no training/promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--delivery',type=Path,required=True);args=ap.parse_args()
    root=args.delivery.resolve();destination=root/'submission_code.zip'
    if destination.exists():raise RuntimeError('Refuse overwrite of already packaged delivery')
    verification=json.loads((root/'models/evidence/packaged_fold5_verification.json').read_text())
    assert verification['packaged_predictions_exactly_equal_saved'] and verification['rows']==144665
    summary_path=root/'models/delivery_summary.json';summary=json.loads(summary_path.read_text())
    assert verification['scores']==summary['validation_fold5_scores']
    summary.pop('standalone_verification',None)
    summary['packaged_fold5_reproduced']=True
    summary['test_generation']='predict.py CLI; actual execution report accompanies result.json outside code ZIP'
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    included=[root/name for name in ['README.md','predict.py','requirements.txt','runtime_versions.json']]
    included += [p for folder in ['src','models'] for p in (root/folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='MANIFEST.json']
    manifest={p.relative_to(root).as_posix():sha(p) for p in included}
    (root/'models/MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    included.append(root/'models/MANIFEST.json')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in included:archive.write(p,p.relative_to(root).as_posix())
    extracted=root/'_verification/extracted_submission_code'
    assert not extracted.exists()
    with zipfile.ZipFile(destination) as archive:archive.extractall(extracted)
    assert all(sha(extracted/name)==digest for name,digest in manifest.items())
    print(json.dumps({'code_zip':str(destination),'code_zip_sha256':sha(destination),
                      'bytes':destination.stat().st_size,'files':len(included),'fresh_extraction':str(extracted)},indent=2))


if __name__=='__main__':main()
