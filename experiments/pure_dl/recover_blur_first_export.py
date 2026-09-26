"""Export completed training checkpoint after disk-full failure; zero updates."""
import sys,json,hashlib
from pathlib import Path
source=Path(__file__).with_name('train_blur_first_sequence.py').resolve()
out=Path('outputs/pure_dl_blur_first_sequence')
assert (out/'fold_1_best_checkpoint.pt').exists() and not (out/'fold_1.pt').exists()
before=hashlib.sha256((out/'fold_1_best_checkpoint.pt').read_bytes()).hexdigest()
# Preserve the incomplete archive as evidence, on the relocated C volume.
partial=out/'fold_1_distributions.npz'
if partial.exists():partial.rename(out/'incomplete_distributions_diskfull.npz')
text=source.read_text()
begin=text.index('        history = []; torch.cuda.reset_peak_memory_stats()')
end=text.index('        stopped_epoch=epoch+1',begin)
replacement="""        validation_history=json.loads((args.out/'fold_1_validation_history.json').read_text())
        assert len(validation_history)==55
        best=torch.load(args.out/'fold_1_best_checkpoint.pt',map_location='cpu',weights_only=False)
        best_epoch=best['epoch'];best_score=best['score'];best_state=best['state_dict']
        assert best_score==max(h['official_total'] for h in validation_history)
        model.load_state_dict(best_state)
        torch.cuda.reset_peak_memory_stats()
        pack=evaluate_current()
        assert abs(base.metric(truth[validation],pred[validation])['TOTAL']-best_score)<1e-10
        best_predictions=pred[validation].copy()
        history=[];epoch=54;outputs=None;loss=None
        print('RECOVERY_EXPORT_ONLY_ZERO_OPTIMIZER_UPDATES',flush=True)
"""
text=text[:begin]+replacement+text[end:]
assert 'opt.step()' not in text
sys.argv=[str(source),'--out',str(out),'--learning-rate','0.001','--schedule','cosine']
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
assert hashlib.sha256((out/'fold_1_best_checkpoint.pt').read_bytes()).hexdigest()==before
report=json.loads((out/'fold_1_report.json').read_text())
report['recovery']={'reason':'disk full during distribution archive write, after original best-checkpoint replay assertions passed','optimizer_updates':0,'best_checkpoint_sha256_unchanged':before,'seconds_field_scope':'recovery preprocessing and export only; original epoch55 log recorded245.0 seconds','original_epoch_predictions_not_persisted':True,'recovery_predictions_twice_identical':True}
(out/'fold_1_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
screen=json.loads((out/'screening_report.json').read_text());screen['folds']=[report];screen['recovery']=report['recovery'];(out/'screening_report.json').write_text(json.dumps(screen,indent=2),encoding='utf-8')
print('RECOVERY_COMPLETE_BEST_CHECKPOINT_UNCHANGED',flush=True)
