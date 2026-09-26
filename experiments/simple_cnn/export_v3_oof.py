"""Run on original v3 server to export only the aligned final OOF needed locally."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();cache=a.root/'cache'
meta=json.loads((cache/'meta.json').read_text())
pred=np.load(a.root/'outputs/assembly_D/final_oof.npy')
truth=np.load(cache/'tr_y.npy');fold=np.load(cache/'tr_fold.npy');well=np.load(cache/'tr_well.npy');depth=np.load(cache/'tr_depth.npy')
assert pred.shape==truth.shape and np.isfinite(pred).all()
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out,pred=pred,truth=truth,fold=fold,well=well,depth=depth,
                    well_ids=np.array(meta['train_wells']),provenance=np.array('assembly_D/final_oof.npy: nested fusion on cached OOF'))
print(a.out,'rows',len(pred),'bytes',a.out.stat().st_size)
