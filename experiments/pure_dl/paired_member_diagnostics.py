"""Read-only bounded redundancy diagnostics; no decoding or model inference."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import expit

source = Path('outputs/pure_dl_paired_backbone_sequence/fold_5_distributions.npz')
out = Path('outputs/paired_member_diagnostics')
out.mkdir(exist_ok=True)
start = time.perf_counter()
with np.load(source) as data:
    indices = data['indices']
    gate = expit(data['member_gate_logits'].astype(np.float64))
    channels = {}
    for j, name in enumerate(('POR','PERM','SW')):
        a, b = gate[:,0,j], gate[:,1,j]
        channels[name] = {
            'member_default_probability_correlation': float(np.corrcoef(a,b)[0,1]),
            'mean_absolute_probability_difference': float(np.abs(a-b).mean()),
            'median_absolute_probability_difference': float(np.median(np.abs(a-b))),
            'member_mean_default_probability': [float(a.mean()),float(b.mean())]}
    available = list(data.files)
result = {'source':str(source),'rows':len(indices),'channels':channels,
          'available_arrays':available,'scope':'gate probabilities only, not decoded predictions or errors',
          'full_decode_not_run':'No member-specific decoded predictions or member CDFs stored. Six 144665x257 by 257x~801 utility products require approximately179 billion multiply-adds plus CDF reconstruction; outside bounded CPU diagnostic budget. Saved component means/std/weights permit future decoding, with float16 quantization caveat.',
          'cpu_seconds':time.perf_counter()-start,'gpu_used':False}
(out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
