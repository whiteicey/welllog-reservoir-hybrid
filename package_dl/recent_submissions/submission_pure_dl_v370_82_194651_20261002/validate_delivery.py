"""Check submission schema, population, order, ranges and optional file manifest."""
from pathlib import Path
import argparse,hashlib,json,math,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from src.data import read_curve

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir','--data-dir',type=Path,required=True)
    ap.add_argument('--result',type=Path,default=ROOT/'result.json');a=ap.parse_args()
    directory=a.data_dir/'test' if (a.data_dir/'test').is_dir() else a.data_dir
    files=sorted(directory.glob('*.txt'));assert files,'No input wells'
    result=json.loads(a.result.read_text(encoding='utf-8'));assert set(result)=={'resultData'}
    assert len(result['resultData'])==len(files);total=0
    for path,item in zip(files,result['resultData']):
        assert set(item)=={'logId','predictions'} and item['logId']==path.stem
        frame=read_curve(path);rows=item['predictions'];assert len(rows)==len(frame)
        for row,depth in zip(rows,frame['DEPTH'].to_numpy()):
            assert set(row)=={'depth','POR','PERM','SW'} and abs(row['depth']-depth)<=1e-6
            for c,lo,hi in [('POR',0,50),('PERM',.01,10000),('SW',8.3,99.9)]:
                value=row[c];assert isinstance(value,(int,float)) and math.isfinite(value) and lo-1e-9<=value<=hi+1e-9
        total+=len(rows)
    zpath=a.result.parent/'result.zip'
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:assert z.namelist()==['result.json'] and z.read('result.json')==a.result.read_bytes()
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps(dict(passed=True,wells=len(files),rows=total,result_schema_and_depth_order=True,
                         result_zip_bytes_match=zpath.exists(),manifest_verified=manifest.exists(),test_accuracy='unknown')))

if __name__=='__main__':main()
