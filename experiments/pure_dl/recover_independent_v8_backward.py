"""Explicit retry from unchanged epoch0 after native CUDA interpolation error."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
from launch_independent_channels_v8 import ROOT,STUDY,PREFIX,trainers


def main():
    sys.stdout.reconfigure(encoding='utf-8');assert not trainers()
    launches=[]
    for key in ('POR_dual','SW_dual'):
        out=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        failure=json.loads((out/'failure.json').read_text())
        assert 'upsample_linear1d_backward_out_cuda' in failure['error']
        assert json.loads((out/'validation_history.json').read_text())==[]
        saved=out/'failed_attempt2';saved.mkdir(exist_ok=False)
        launch=json.loads((out/'launch.json').read_text())
        for name in ('config.json','failure.json','runtime_status.json','initial_verification.json','model_best.pt','last_state.pt','channel_pred.npy','validation_history.json','train.log','train.err','launch.json'):
            p=out/name
            if p.exists():p.rename(saved/name)
        note={'stage':'first backward, before optimizer step','previous_attempt':2,
              'fix':'deterministic warn_only for native CUDA interpolate backward; all6arms use same setting',
              'bitwise_training_reproduction':False,'error_evidence':'failed_attempt2/failure.json'}
        (out/'backward_failure_attempt2.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            p=subprocess.Popen(launch['command'],cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={**launch,'pid':p.pid,'attempt':3,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');launches.append(row)
        print('EXPLICIT_RETRY',json.dumps(row,ensure_ascii=False),flush=True)
    (STUDY/'backward_retry.json').write_text(json.dumps(launches,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## CUDA反向兼容性修复\n首批两模型初始目标预测逐元素复现，但严格确定性在首次linear插值反向报错，未执行optimizer.step。失败attempt2完整保存在各目录failed_attempt2。六项统一采用deterministic warn_only保留可用确定性算子，对CUDA插值反向允许警告；固定随机种子仍不能保证训练逐位重现。该设置已写入config。旧源代码未改，明确从原起点重新启动首批，不冒称续跑。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
