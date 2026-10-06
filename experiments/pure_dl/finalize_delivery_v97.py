"""New-directory submission packaging and bounded standalone verification."""
from pathlib import Path
import argparse,ast,hashlib,importlib,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
D=ROOT.parent/'submission_pure_dl_v97_81_891019_20260930'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def files():
    return sorted(p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '_verification' not in p.parts
                  and p.suffix not in ('.pyc','.zip') and p.name not in ('result.json','result.inference.json','delivery_validation.json','MANIFEST.json'))

def prepare():
    source=ROOT/'experiments/pure_dl';names={Path(__file__).name,'verify_delivery_graph_v97.py','train_root_fullcv_v97.py',
       'train_paired_local_context_router80_sequence.py','prepare_root_fullcv_v97.py','prepare_fullcv_v97.py',
       'run_fullcv_v97.py','verify_decoder_fullcv_v97.py','record_fullcv_v97.py'}
    pending=list(names)
    while pending:
        name=pending.pop();path=source/name
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            mods=[n.name.split('.')[0] for n in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for m in mods:
                local=m+'.py'
                if (source/local).exists() and local not in names:names.add(local);pending.append(local)
    for name in names:shutil.copy2(source/name,D/'experiments/pure_dl'/name)
    for f in range(1,5):
        directory=ROOT/f'outputs/pure_dl_fullcv_v97/fold_{f}'
        for p in directory.glob('*/config.json'):
            dest=D/'models/training_evidence'/f'fold_{f}'/p.parent.name;dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest/p.name)
            report=p.parent/'report.json'
            if report.exists():shutil.copy2(report,dest/report.name)
        for name in ('data_manifest.json','fold_report.json','disk_replay_proof.json'):
            shutil.copy2(directory/name,D/'models/training_evidence'/f'fold_{f}'/name)
    runtime=json.loads((D/'runtime_versions.json').read_text());runtime['psutil']=importlib.import_module('psutil').__version__
    write(D/'runtime_versions.json',runtime)
    with (D/'requirements.txt').open('a',encoding='utf-8') as s:s.write('psutil=='+runtime['psutil']+'\n')
    write(D/'models/training_source_manifest.json',{str(p.relative_to(D)):sha(p) for p in (D/'experiments/pure_dl').glob('*.py')})
    print('PREPARED',len(list((D/'experiments/pure_dl').glob('*.py'))),flush=True)

def candidate():
    proof=json.loads((D/'models/packaged_oof_verification.json').read_text());assert proof['all723556_packaged_predictions_exact']
    dest=D/'_verification/code_candidate.zip';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise RuntimeError('Candidate already exists')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files():z.write(p,p.relative_to(D).as_posix())
    extracted=D/'_verification/extracted_candidate';extracted.mkdir()
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None;z.extractall(extracted)
    for p in files():assert sha(p)==sha(extracted/p.relative_to(D))
    print('CANDIDATE_EXTRACTED',extracted,flush=True)

def finish():
    x=D/'_verification/extracted_candidate';actual=x/'_run/result.json'
    assert actual.read_bytes()==(D/'result.json').read_bytes(),'Independent CLI test result changed'
    audit=json.loads((x/'_run/standalone_audit.json').read_text());assert audit['passed']
    write(D/'standalone_execution_verification.json',audit|dict(all_test_result_bytes_exact=True,result_sha256=sha(actual),
          inference=json.loads(actual.with_suffix('.inference.json').read_text()),scope='fresh extracted candidate; original workspace and training-data reads blocked'))
    (D/'PROJECTS.md').write_text('# 目标与进度\n\n目标：交付冻结v97完整五折模型和测试提交结果。\n\n已完成：五折推理图、必要源码与配置、10口井95,948行结果、723,556点导出OOF逐元素复现、独立解压CLI全测试井运行和结果逐字节一致、无原工作区或训练数据读取审计。\n\n计划/未完成：本提交包任务无剩余工作；原完整OOF>82目标尚未达成。\n\n后续可选工作：返回研究目录落实用户要求的包含来源预训练在内最多五阶段的新配方；本包和历史交付均保持冻结。\n',encoding='utf-8')
    included=files();manifest={p.relative_to(D).as_posix():sha(p) for p in included}
    write(D/'MANIFEST.json',manifest);included.append(D/'MANIFEST.json')
    destination=D/'submission_code.zip'
    if destination.exists():raise RuntimeError('Refuse replacing a finalized code ZIP')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in included:z.write(p,p.relative_to(D).as_posix())
    final=D/'_verification/extracted_final'
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None;z.extractall(final)
    assert all(sha(final/n)==v for n,v in manifest.items())
    # No runtime code, dependencies or model files changed after the executed
    # extraction. Only the progress record and verification evidence were added.
    runtime_files=['predict.py','requirements.txt','runtime_versions.json']+[p.relative_to(D).as_posix() for folder in ('experiments','package')
        for p in (D/folder).rglob('*') if p.is_file() and p.suffix=='.py']+[f'models/fold_{f}.pt' for f in range(1,6)]
    assert all(sha(final/n)==sha(x/n) for n in runtime_files)
    result=json.loads((D/'result.json').read_text());rows=sum(len(e['predictions']) for e in result['resultData'])
    assert len(result['resultData'])==10 and rows==95948
    with zipfile.ZipFile(D/'result.zip') as z:assert z.namelist()==['result.json'] and z.read('result.json')==(D/'result.json').read_bytes()
    proof=json.loads((D/'models/packaged_oof_verification.json').read_text())
    validation=dict(passed=True,wells=10,rows=rows,all723556_oof_predictions_exact=proof['all723556_packaged_predictions_exact'],
       official_oof_scores=proof['original_oof_scores'],manifest_all_hashes_match=True,final_zip_crc_passed=True,
       final_extracted_runtime_matches_executed_candidate=True,standalone_cli_all_test_result_bytes_exact=True,
       original_workspace_and_train_reads_blocked=True,result_zip_json_bytes_match=True,
       result_sha256=sha(D/'result.json'),result_zip_sha256=sha(D/'result.zip'),code_zip_sha256=sha(destination),
       code_zip_bytes=destination.stat().st_size,code_files=len(included),test_accuracy='unknown; test labels unavailable')
    write(D/'delivery_validation.json',validation)
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as s:s.write('\n\n## v97提交包交付\n已在新目录 `submission_pure_dl_v97_81_891019_20260930` 完成结果ZIP、独立代码/五折模型ZIP、OOF81.891019逐元素复现及独立解压全部测试井推理验证。历史交付不改。本包仍为来源预训练+18后续阶段，未声称精简。下一步待做：用户要求的最多五个总训练阶段新配方；先第五折比较，完整OOF>82仍未达成。\n')
    print('DELIVERY_FINAL',json.dumps(validation),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','candidate','finish'));a=ap.parse_args()
    globals()[a.action]()
