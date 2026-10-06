import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/training_matmul_timing_v122_20261001'
    if not study.exists():return
    p=study/'timing.json';v=json.loads(p.read_text()) if p.exists() else {}
    text='Prepared synthetic timing, no trained model/accuracy' if not v else 'TIMING ONLY: IEEE '+str(v['IEEE']['forward_backward_ms'])+'ms vs TF32 '+str(v['TF32']['forward_backward_ms'])+'ms; no model/accuracy score. No epoch speedup evidence.'
    node('timing_probe:v122','v122 utility matmul timing only','8192x257 by257x801 synthetic own training utility forward/backward,8 warmup32 timed each, while1SW task active. No labels/model update or decoder precision change. Small absolute saving; no full training precision trial.',text,stage='optimization',status='recorded' if v else 'prepared',evidence=['results/deep_learning/training_matmul_timing_v122_20261001/README.md'],resultText=text)
    edge('pure_dl_compact_sw_readout_split_v121_fold5/split','timing_probe:v122','comparison','Runtime-only timing alongside active SW, no parameter lineage or accuracy comparison')
