"""Explicit original-fold5 data preparation in the NEW v16 study directory.

The v14 preparation implementation is reused unchanged. It refits on the
original64 training wells; no inner48 trained model/statistics are loaded.
"""
from pathlib import Path
import json
import sys
import numpy as np
import prepare_sw_group_v14 as preparation

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/latent_fold5_v16_20260928'


def main():
    if len(sys.argv) != 1:
        raise RuntimeError('This entry point only prepares the original fifth fold')
    preparation.STUDY = STUDY
    sys.argv = [sys.argv[0], '--phase', 'outer']
    preparation.main()
    manifest_path = STUDY/'data_outer_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert len(manifest['fit_wells']) == 64 and len(manifest['monitor_wells']) == 16
    assert manifest['fit_valid_points'] == 578891 and manifest['monitor_valid_points'] == 144665
    with np.load(STUDY/'data_outer.npz') as data, np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as ref:
        rows = data['monitor_raw']
        mask = ref['fold'] == 4
        assert np.array_equal(data['truth'][rows], ref['truth'][mask, 2])
        assert np.array_equal(data['wells'][rows], ref['well'][mask])
        assert not data['fitmask'][rows].any()
    manifest.update(study='v16 originalfold5', inner_split=None,
        outer_fifth_used_for_fitting=False, outer_fifth_used_for_checkpoint_selection=True,
        original_validation_row_order_verified=True, reused_inner_checkpoint=False,
        note='Legacy fit_or_selection flag is true because fold5 selects checkpoints, never because it fits preprocessing or gradients.')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ORIGINAL_FIFTH_ORDER_VERIFIED',144665)


if __name__ == '__main__':
    main()
