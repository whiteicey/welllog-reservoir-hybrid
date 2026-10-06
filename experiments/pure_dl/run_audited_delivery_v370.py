"""Execute the extracted CLI, forbidding original workspace/training reads."""
from pathlib import Path
import json,os,runpy,sys,time
import torch
REPO=Path(__file__).resolve().parents[2]
D=REPO.parent/'submission_pure_dl_v370_82_194651_20261002';X=D/'_verification/extracted_candidate';TEST=REPO.parent/'data/test'
output=X/'_run/result.json';output.parent.mkdir(exist_ok=True)
workspace=os.path.normcase(str(REPO.parent.resolve()))+os.sep
allow=[os.path.normcase(str(p.resolve()))+os.sep for p in (X,TEST)]
reads=set();started=time.perf_counter()
def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
    if not path.startswith(workspace):return
    if not any(path.startswith(prefix) for prefix in allow):raise RuntimeError('Standalone forbidden workspace access: '+path)
    if path.startswith(allow[1]):reads.add(path)
sys.addaudithook(audit)
sys.argv=[str(X/'predict.py'),'--data_dir',str(TEST),'--output',str(output),'--device','cuda']
runpy.run_path(str(X/'predict.py'),run_name='__main__')
report=dict(passed=True,original_workspace_reads_blocked=True,training_data_reads_blocked=True,allowed_input_well_files=len(reads),
            seconds=time.perf_counter()-started,extracted_directory=str(X),input_directory=str(TEST),command=sys.argv)
assert len(reads)==10
(X/'_run/standalone_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('STANDALONE_AUDITED',json.dumps(report),flush=True)
