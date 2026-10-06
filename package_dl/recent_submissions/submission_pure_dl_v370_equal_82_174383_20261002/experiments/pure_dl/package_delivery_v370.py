"""Prepare, extract, and finalize the new submission with concrete readback."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
D=ROOT.parent/'submission_pure_dl_v370_82_194651_20261002'
S=ROOT/'results/deep_learning/two_seed_fullcv_v370_20261002'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def included():
    return sorted(p for p in D.rglob('*') if p.is_file() and '_verification' not in p.parts and '__pycache__' not in p.parts
                  and p.suffix not in ('.pyc','.zip') and p.name not in ('MANIFEST.json','delivery_validation.json'))

def prepare():
    summary=read(D/'models/delivery_summary.json')
    shutil.copy2(ROOT.parent/'submission_pure_dl_v97_81_891019_20260930/validate_delivery.py',D/'validate_delivery.py')
    for filename in ('build_delivery_v370.py','delivery_predict_v370.py','verify_delivery_v370.py','package_delivery_v370.py','run_audited_delivery_v370.py'):
        shutil.copy2(ROOT/'experiments/pure_dl'/filename,D/'experiments/pure_dl'/filename)
    weights='\n'.join('| '+str(f['fold'])+' | '+' | '.join(f"{100*f['weights'][t]:.0f}%/{100*(1-f['weights'][t]):.0f}%" for t in ('POR','PERM','SW'))+' |' for f in summary['folds'])
    scores=read(S/'FINAL.json')
    table='\n'.join(f"| {f['fold']} | {f['fixed_scores']['TOTAL']:.6f} | {f['diagnostic_scores']['TOTAL']:.6f} |" for f in scores['folds'])
    readme=f'''# V370 两种子五折 submission

用户指定的版本：逐折逐头最优配比，完整分组OOF **82.19465144853783**。
该分数使用各折验证标签选择配比，是调参诊断，不是无偏验证或未知测试集精度。
第五折选择固定配对和固定比例的对照OOF为 **82.16741757218335**。

## 文件

- `result.zip`：实际测试集提交文件，仅含 `result.json`。
- `submission_code.zip`：独立推理代码、必要源码、五折/两种子完整推理模型、配置和核验记录。
- `models/fold_1.pt` 至 `fold_5.pt`：每折包含原模型/新种子的三个输出模型及本折预处理，无需研究工作区或训练数据。
- `models/delivery_summary.json`：模型SHA、逐头比例、来源及完整评分。
- `models/packaged_oof_verification.json`、`standalone_execution_verification.json`、`delivery_validation.json`：最终验证结果。

## 推理

已验证环境：Python3.10.11、torch2.7.0+cu128、NumPy1.26.4、pandas2.2.3，详见 `runtime_versions.json`。
在已有 normal 环境下可直接运行。其他环境安装 `requirements.txt`；GPU版PyTorch需匹配CUDA驱动。

```powershell
python predict.py --data_dir D:\\match_predict\\data --output result.json --device cuda
python validate_delivery.py --data_dir D:\\match_predict\\data --result result.json
```

`--data_dir` 接受含test子目录的数据根目录，或测试TXT文件目录。默认 `--weight-policy diagnostic` 为用户指定版本。
`--weight-policy fixed` 使用原模型POR56%/PERM44%/SW68%的固定对照。
默认POR/PERM32窗口、SW16窗口，原256/1024长度、stride128、256概率格、800连续动作加默认动作不变；跨井批次组织保留。

## 每折原模型/新种子比例

| 折 | POR | PERM | SW |
|---|---|---|---|
{weights}

每头每折所有点使用同一个比例，不按行、默认/连续真值类别或井标签选择模型。
先在每折内融合两种子的联合概率分布，再对五折期望效用等权平均，最后统一argmax动作；不平均解码后的预测值。
测试集按全部井输入组织跨井批次，井边界不互通。各折统计只使用该折训练井拟合值，测试输入仅transform。

## 已有评分

| 折 | 固定比例对照 | 本包逐头最优诊断 |
|---|---:|---:|
{table}
| 完整OOF | 82.167418 | 82.194651 |

本包诊断OOF：POR79.878340493724 / PERM90.764266873833 / SW75.610445413083。
原始GroupKFold五折、80井、723556验证点。两种子各8阶段，每折配对合计16个历史训练阶段；融合不新增训练阶段。
来源是本折训练的双专家骨干及POR341/PERM336/SW351增补分支，三个头各一个联合骨干/读出新增阶段。
第五折复用历史已验证模型，历史耐心配置与后续补折不同，留存原配置。

## 范围

完整推理图保留已训练hooks和边界机制；`experiments/pure_dl` 是架构依赖及来源源码留证。
留存训练脚本原配置/路径语义，不声称在本交付目录中一键重训。推理入口已独立解压运行验证。
没有测试标签，测试结果只验证井ID、95948行、深度顺序、字段、有限值和数值范围，不能据此报告测试精度。
旧submission目录保持不变。本目录不启动训练。
'''
    (D/'README.md').write_text(readme,encoding='utf-8')
    requirements=(D/'requirements.txt').read_text(encoding='utf-8')
    if '--extra-index-url' not in requirements:(D/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu128\n'+requirements,encoding='utf-8')
    for f in summary['folds']:
        for source in f['source_models']:
            cp=ROOT/source['checkpoint'];dest=D/'models/training_evidence'/f"fold_{f['fold']}"/source['identity']/source['target']
            for filename in ('config.json','report.json'):
                p=cp.parent/filename
                if p.exists():dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/filename)
    write(D/'models/packaged_source_manifest.json',{p.relative_to(D).as_posix():sha(p) for folder in ('experiments','package') for p in (D/folder).rglob('*.py')})
    print('PREPARED',str(D),flush=True)

def candidate():
    proof=read(D/'models/packaged_oof_verification.json');assert proof['passed']
    dest=D/'_verification/code_candidate.zip';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in included():z.write(p,p.relative_to(D).as_posix())
    extracted=D/'_verification/extracted_candidate';assert not extracted.exists()
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None;z.extractall(extracted)
    for p in included():assert sha(p)==sha(extracted/p.relative_to(D))
    print('CANDIDATE_EXTRACTED',str(extracted),flush=True)

def finish():
    X=D/'_verification/extracted_candidate';result=X/'_run/result.json';audit=read(X/'_run/standalone_audit.json')
    assert audit['passed'] and result.read_bytes()==(D/'result.json').read_bytes()
    write(D/'standalone_execution_verification.json',audit|dict(all_test_result_bytes_exact=True,result_sha256=sha(result),inference=read(result.with_suffix('.inference.json'))))
    proof=read(D/'models/packaged_oof_verification.json');assert proof['passed']
    (D/'PROJECTS.md').write_text('# 目标与进度\n\n目标：交付用户指定82.194651版本submission。\n\n已完成：10套推理图、逐折逐头配比、完整OOF重放、10井95948行实际测试结果、独立解压CLI全测试推理、原工作区/训练数据读取阻断、字段/深度/范围/ZIP核验。\n\n计划/未完成：本提交包任务无剩余工作。82.194651是调参诊断，未知测试精度未验证；每模型8阶段、配对16历史阶段。\n\n后续可选：独立泛化、统一阶段压缩消融，需在研究目录另行开展；旧交付冻结。\n',encoding='utf-8')
    manifest={p.relative_to(D).as_posix():sha(p) for p in included()};write(D/'MANIFEST.json',manifest)
    dest=D/'submission_code.zip';assert not dest.exists()
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in included()+[D/'MANIFEST.json']:z.write(p,p.relative_to(D).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    runtime=['predict.py','requirements.txt']+[p.relative_to(D).as_posix() for folder in ('experiments','package') for p in (D/folder).rglob('*.py')]+[f'models/fold_{f}.pt' for f in range(1,6)]
    assert all(sha(D/p)==sha(X/p) for p in runtime),'Final runtime must equal executed extracted runtime'
    data=read(D/'result.json');assert len(data['resultData'])==10 and sum(len(w['predictions']) for w in data['resultData'])==95948
    with zipfile.ZipFile(D/'result.zip') as z:assert z.namelist()==['result.json'] and z.read('result.json')==(D/'result.json').read_bytes()
    validation=dict(passed=True,wells=10,rows=95948,packaged_oof_verified=proof['passed'],packaged_oof_scores=proof['scores'],
        independent_extracted_CLI_test_result_bytes_exact=True,original_workspace_and_training_reads_blocked=True,
        final_code_zip_CRC_and_manifest_passed=True,final_runtime_exact_to_executed_candidate=True,result_zip_json_bytes_exact=True,
        result_sha256=sha(D/'result.json'),code_zip_sha256=sha(dest),code_zip_bytes=dest.stat().st_size,
        result_zip_sha256=sha(D/'result.zip'),score_scope='Foldwise per-head ratio tuning diagnostic, not unbiased validation',test_accuracy='unknown')
    write(D/'delivery_validation.json',validation)
    note='\n\n2026-10-02 用户指定82.194651诊断版本submission已交付到新目录 `submission_pure_dl_v370_82_194651_20261002`。完整推理图、各折逐头比例、result.zip及submission_code.zip；完整OOF模型重放和独立解压10井95948行推理/字节一致/读取隔离/ZIP核验通过。旧交付不改，每模型8阶段、配对16历史阶段，测试精度未知。结果见新目录delivery_validation.json。\n'
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',S/'PROJECTS.md',S/'README.md'):
        with p.open('a',encoding='utf-8') as stream:stream.write(note)
    print('DELIVERY_FINAL',json.dumps(validation),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','candidate','finish'));args=ap.parse_args();globals()[args.action]()
