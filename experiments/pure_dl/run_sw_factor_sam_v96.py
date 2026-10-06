"""Explicit one-arm serial launch, no queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_factor_sam_v96 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('Explicit serial study: a trainer is active')
    out=ROOT/f'outputs/pure_dl_sw_factor_sam_v96_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_factor_sam_v96.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v96 '+a.mode+'显式启动'
    for p in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(p,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_factor_sam_v96_20260930','- [v96 固定SAM默认atom/私有密度优化](results/deep_learning/sw_factor_sam_v96_20260930/README.md)：从v95完整函数相同热启动，likelihood/joint同SAM.1；尚无新训练收益。')
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v96固定atom私有密度范围','## v96固定atom私有密度范围\n'+(STUDY/'AGENTS.md').read_text(encoding='utf-8'))
    print(json.dumps(launch),flush=True)
if __name__=='__main__':main()
