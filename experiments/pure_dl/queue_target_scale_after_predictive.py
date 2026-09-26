"""One-shot serial handoff after the currently running predictive screen/audits."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys

parent_pid = 5048
expected = 'run_paired_predictive_screen.py'
command = subprocess.check_output([
    'powershell', '-NoProfile', '-Command',
    f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {parent_pid}').CommandLine"
], text=True)
assert expected in command, 'Expected live predecessor runner required; do not guess or restart'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
handle = kernel.OpenProcess(0x00100000 | 0x1000, False, parent_pid)
assert handle, 'Cannot acquire predecessor process handle'
print('WAIT_FOR_VERIFIED_PREDECESSOR', parent_pid, flush=True)
try:
    while True:
        state = kernel.WaitForSingleObject(handle, 30000)
        if state == 0:
            break
        assert state == 258, 'Unexpected process wait error'
    code = wintypes.DWORD()
    assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    assert code.value == 0, 'Predecessor failed/stopped; no automatic next training'
finally:
    kernel.CloseHandle(handle)

previous = Path('outputs/pure_dl_paired_predictive_sequence')
for name in ['screening_verification.json', 'fifth_fold_verification.json',
             'global_point_table_verification.json', 'conflict_trace_verification.json']:
    assert json.loads((previous/name).read_text())['passed'], name
proof = json.loads((previous/'screening_verification.json').read_text())
if proof['qualifies_for_remaining_folds']:
    print('PREDECESSOR_PASSED_802: return to root for full-CV decision', flush=True)
    sys.exit(0)
out = Path('outputs/pure_dl_target_scale_readout_sequence')
assert not (out/'train.log').exists(), 'Never overwrite or restart an existing trial'
for name in ['cpu_preflight.json', 'gpu_preflight.json']:
    assert json.loads((out/name).read_text())['passed']
(out/'serial_handoff.json').write_text(json.dumps({
    'predecessor_pid': parent_pid, 'predecessor_verified_score': proof['screening_fold_score'],
    'reason': 'Verified predecessor below80.2; launch prepared independent structure',
    'automatic_remaining_folds': False
}, indent=2))
subprocess.run([sys.executable, '-u', 'experiments/pure_dl/run_target_scale_readout_screen.py'], check=True)
