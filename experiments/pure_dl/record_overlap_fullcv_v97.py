"""Keep authoritative batch records accurate for source/main-stage overlap."""
from pathlib import Path
from contextlib import contextmanager
import json
import time
from fullcv_context_v97 import ROOT, STUDY
import record_fullcv_v97


@contextmanager
def coherent_json_reads():
    """Tolerate a live root's non-atomic progress write, without retrying training."""
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.suffix != '.json':
            return original(path, *args, **kwargs)
        for attempt in range(60):
            try:
                text = original(path, *args, **kwargs)
                json.loads(text)
                return text
            except (OSError, json.JSONDecodeError):
                if attempt == 59:
                    raise
                time.sleep(.05)

    Path.read_text = read_text
    try:
        yield
    finally:
        Path.read_text = original


def main():
    with coherent_json_reads():
        record_fullcv_v97.main()
    status=json.loads((STUDY/'controller_status.json').read_text(encoding='utf-8'))
    tail=(STUDY/'tail_admission_plan.json').exists()
    admission=('来源最低显示空闲6.8GiB，普通阶段4GiB；已部署的最后四个SW阶段在无worker时使用显示3GiB/CUDA4GiB双门槛，来源与实测见TAIL_ADMISSION.md。'
               if tail else '来源最低空闲6.8GiB，阶段最低4GiB。')
    for name in ('PROJECTS.md','PROGRESS.md'):
        path=STUDY/name;text=path.read_text(encoding='utf-8')
        text=text.replace('串行控制器逐项执行','主阶段依序执行，来源预训练与下一折重叠')
        text=text.replace('串行并发1、上限2','最多2个GPU任务，最多1个SW任务；主阶段与下一折来源预训练重叠')
        text+='\n\n当前调度：'+str(status.get('execution'))+'；CPU调度器接管保持原训练PID，不修改训练源码/配置/权重。'+admission+'工作量与同配方原五折一致。重叠事件及完整授权批次墙钟留档，不将单项时长相加当加速证据。失败停止且不重试。\n'
        path.write_text(text,encoding='utf-8')
    for project in (ROOT.parent/'PROJECTS.md', ROOT/'PROJECTS.md'):
        with project.open('a',encoding='utf-8') as stream:
            stream.write('\n\n'+(STUDY/'PROGRESS.md').read_text(encoding='utf-8'))
    print('OVERLAP_RECORDS_UPDATED')


if __name__=='__main__':main()
