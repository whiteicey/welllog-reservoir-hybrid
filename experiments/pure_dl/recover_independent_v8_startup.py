"""One explicit retry of two pre-data startup failures. Preserves first attempt."""
from pathlib import Path
from datetime import datetime
import json,sys,subprocess
from launch_independent_channels_v8 import ROOT,STUDY,PREFIX,trainers

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    assert not trainers(),'Do not touch active trainers'
    retry=[]
    for key in ('POR_dual','SW_dual'):
        out=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        assert not (out/'config.json').exists() and not (out/'model_best.pt').exists()
        error=(out/'train.err').read_text(encoding='utf-8')
        assert "NameError: name 'sys' is not defined" in error
        assert not (out/'startup_failure_attempt1.json').exists(),'Retry already attempted'
        old=json.loads((out/'launch.json').read_text())
        proof={'launch':old,'stage':'before data loading or training','error':error,'fix':'add missing sys import','observed_at':datetime.now().astimezone().isoformat()}
        (out/'startup_failure_attempt1.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'train.err').rename(out/'train.attempt1.err')
        (out/'train.log').rename(out/'train.attempt1.log')
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(old['command'],cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        new={**old,'pid':p.pid,'attempt':2,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
        retry.append(new);print('EXPLICIT_RETRY',json.dumps(new,ensure_ascii=False),flush=True)
    (STUDY/'startup_retry.json').write_text(json.dumps(retry,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## 启动失败及修复\n首批POR_dual/SW_dual在main开头因遗漏sys导入失败，未加载数据或训练；已保存startup_failure_attempt1.json和原err。修复导入并明确重试，非自动恢复。另一次只读shell命令转义失败，不影响训练。结果以第二次实际日志为准。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
