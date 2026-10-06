"""Create a new explicit-fold source trainer; historical trainer stays frozen."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT/'results/deep_learning/fullcv_v97_20260930'


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source block: {old[:80]!r}, got {text.count(old)}')
    return text.replace(old, new)


def main():
    original = HERE/'train_paired_local_context_router80_sequence.py'
    s = original.read_text(encoding='utf-8')
    s = replace(s, 'from training_por_t3_full import TrainingDynamicsProbe',
        'from fullcv_resume_v97 import CompactProbe as TrainingDynamicsProbe, atomic_save, cpu_state, rng_state, restore_rng')
    s = replace(s, 'from fold5_screening_804 import qualifies, THRESHOLD, FOLD_INDEX, fold_plan',
        'from fullcv_context_v97 import STUDY, RECIPE, sha\nTHRESHOLD=80.5\nFOLD_INDEX=4')
    begin = s.index("    ap.add_argument('--fold-count'")
    end = s.index('    torch.set_num_threads(3)', begin)
    s = s[:begin]+'''    ap.add_argument('--fold', type=int, choices=[1,2,3,4], required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    proof = json.loads((STUDY/'promotion.json').read_text(encoding='utf-8'))
    authorization = json.loads((STUDY/'authorization.json').read_text(encoding='utf-8'))
    assert authorization['remaining_folds_authorized'] and authorization['recipe_sha256']==sha(RECIPE)
    assert proof['passed'] and proof['screening_fold']==5 and proof['scores']['TOTAL']>THRESHOLD
    assert proof['recipe_sha256']==sha(RECIPE)
'''+s[end:]
    s = replace(s, "Path(__file__).with_name('fold5_screening_804.py')", "Path(__file__).with_name('fullcv_resume_v97.py')")
    s = replace(s, "'outer_wells':", "'source_role':'fold-local prerequisite, not final three-channel recipe', 'original_fold':args.fold, 'fifth_recipe_sha256':sha(RECIPE), 'historical_trainer_sha256':'"+hashlib.sha256(original.read_bytes()).hexdigest()+"', 'outer_wells':")
    begin=s.index('    for f in fold_plan(args.fold_count):')
    end=s.index('        t0 = time.perf_counter();',begin)
    s=s[:begin]+'''    for f in [args.fold-1]:
'''+s[end:]
    s = replace(s, '        history = []; torch.cuda.reset_peak_memory_stats()',
        '        history = []; torch.cuda.reset_peak_memory_stats()\n        last_path=args.out/f\'fold_{f+1}_last_state.pt\'\n        start_epoch=0; previous_elapsed=0.; previous_peak=0.')
    s = replace(s, "        for epoch in range(cfg['epochs']):", '''        if last_path.exists():
            if not args.resume:raise RuntimeError('Last state exists; an explicit --resume is required')
            saved=torch.load(last_path,map_location='cpu',weights_only=False)
            assert saved['config']==cfg and saved['fold']==f+1
            model.load_state_dict(saved['current_state'])
            opt.load_state_dict(saved['optimizer']);router_opt.load_state_dict(saved['router_optimizer'])
            sched.load_state_dict(saved['scheduler']);router_sched.load_state_dict(saved['router_scheduler'])
            best_state=saved['best_state'];best_predictions=saved['best_predictions']
            best_score=saved['best_score'];best_epoch=saved['best_epoch'];bad_epochs=saved['bad_epochs']
            history=saved['history'];validation_history=saved['validation_history']
            policy.__dict__.update(saved['policy']);table.__dict__.update(saved['table'])
            supervision_weights.copy_(torch.as_tensor(table.weight,device='cuda'))
            assert not table.allowed[valid_to_raw[validation]].any()
            probe.recover(saved['probe'],model,x)
            restore_rng(saved['rng'])
            start_epoch=saved['completed_epoch'];previous_elapsed=saved['elapsed_seconds'];previous_peak=saved['peak_gpu_allocated_mib']
            print('EXPLICIT_RESUME',json.dumps({'fold':f+1,'completed_epoch':start_epoch}),flush=True)
            del saved
        loop_end=start_epoch if start_epoch>=10 and bad_epochs>=18 else cfg['epochs']
        for epoch in range(start_epoch,loop_end):''')
    s = replace(s, "                torch.save({'state_dict':best_state,'epoch':best_epoch,'score':best_score,'config':cfg},args.out/f'fold_{f+1}_best_checkpoint.pt')",
        "                atomic_save({'state_dict':best_state,'epoch':best_epoch,'score':best_score,'config':cfg},args.out/f'fold_{f+1}_best_checkpoint.pt')")
    s = replace(s, "                weight_trace.append(q.astype(np.float32));update_epochs.append(epoch+1)\n                np.savez_compressed(args.out/f'fold_{f+1}_conflict_weights.npz',epochs=np.asarray(update_epochs),raw_ids=probe.raw,weights=np.stack(weight_trace))",
        '                probe.commit(q)')
    s = replace(s, "            if epoch+1>=cfg['early_stopping']['minimum_epochs'] and bad_epochs>=cfg['early_stopping']['patience']:break", '''            atomic_save(dict(config=cfg,fold=f+1,current_state=cpu_state(model),best_state=best_state,
                best_predictions=best_predictions,best_score=best_score,best_epoch=best_epoch,bad_epochs=bad_epochs,
                completed_epoch=epoch+1,history=history,validation_history=validation_history,
                optimizer=opt.state_dict(),router_optimizer=router_opt.state_dict(),scheduler=sched.state_dict(),router_scheduler=router_sched.state_dict(),
                policy=policy.__dict__,table=table.__dict__,probe=probe.recovery_state(),rng=rng_state(),
                elapsed_seconds=previous_elapsed+time.perf_counter()-t0,
                peak_gpu_allocated_mib=max(previous_peak,torch.cuda.max_memory_allocated()/1024**2)),last_path)
            if epoch+1>=cfg['early_stopping']['minimum_epochs'] and bad_epochs>=cfg['early_stopping']['patience']:break''')
    s = replace(s, '        stopped_epoch=epoch+1\n        probe.finalize()',
        "        stopped_epoch=len(history)\n        probe.export_quality(args.out/f'fold_{f+1}_conflict_weights.npz')\n        probe.finalize()")
    s = replace(s, "        torch.save(checkpoint, args.out/f'fold_{f+1}.pt')",
        "        atomic_save(checkpoint, args.out/f'fold_{f+1}.pt')")
    s = replace(s, "              'seconds': time.perf_counter()-t0, 'loss': history, 'all_validation_rows_covered_once': True,\n              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}",
        "              'seconds': previous_elapsed+time.perf_counter()-t0, 'loss': history, 'all_validation_rows_covered_once': True,\n              'peak_gpu_allocated_mib': max(previous_peak,torch.cuda.max_memory_allocated()/1024**2), 'source_role':'prerequisite_only', 'complete_five_folds':False}")
    begin = s.index('    assert np.isfinite(pred[np.isin(fold,fold_plan(args.fold_count))]).all()')
    end = s.index("\n\nif __name__",begin)
    s = s[:begin]+'''    assert np.isfinite(pred[fold==args.fold-1]).all() and np.isnan(pred[fold!=args.fold-1]).all()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(base.ROOT))] for p in sources)
    base.write(args.out/'source_report.json',dict(folds=reports,complete_five_folds=False,
        source_role='own-fold local-context80 prerequisite; downstream stages remain',seconds_this_invocation=time.perf_counter()-start))
'''+s[end:]
    destination=HERE/'train_root_fullcv_v97.py'
    if destination.exists() and destination.read_text(encoding='utf-8')!=s:
        raise RuntimeError('Generated trainer already differs; do not mutate an active source')
    destination.write_text(s,encoding='utf-8')
    authorization=dict(remaining_folds_authorized=True,user_instruction='恢复其余四折训练',
        recipe_sha256=json.loads((STUDY/'promotion.json').read_text())['recipe_sha256'],
        authorized_at=datetime.now(timezone.utc).astimezone().isoformat(),fold_order=[1,2,3,4],reuse_verified_fifth=True,
        maximum_gpu_jobs=2,initial_execution='serial',old_pause_superseded=True)
    if not (STUDY/'authorization.json').exists():
        (STUDY/'authorization.json').write_text(json.dumps(authorization,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## 2026-09-30 用户明确恢复其余四折\n用户答复“恢复其余四折训练”。仅恢复当前v96第五80.504197配方的原其余四折；复用已核验第五预测，各折全部学习依赖/预处理/Q重新使用自身64训练井。旧暂停规则在本范围被此授权覆盖；不启动其他旧队列。首个原local-context80来源训练入口已准备，原子完整last含优化器/调度/RNG/Q策略/观测来源/当前和最佳。初始串行，不改已训练历史源码或交付。剩余下游训练入口尚在实施，完整OOF>82尚未验证。\n'
    for path in [ROOT.parent/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'AGENTS.md',STUDY/'PROJECTS.md',STUDY/'README.md']:
        with path.open('a',encoding='utf-8') as stream:stream.write(note)
    print(json.dumps(dict(trainer=str(destination),historical_source_unchanged=True,authorization=str(STUDY/'authorization.json'))))


if __name__=='__main__':main()
