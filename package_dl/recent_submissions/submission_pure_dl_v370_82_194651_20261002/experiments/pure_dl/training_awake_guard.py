"""Temporary Windows idle-sleep request for a training thread, never a power-plan edit.
Use only in NEW training entrypoints: with TrainingAwake(): main().
Do not alter frozen trained source modules to add this retroactively.
"""
import os
import ctypes
class TrainingAwake:
    def __enter__(self):
        self.active=False
        if os.name=='nt':
            self.set_state=ctypes.windll.kernel32.SetThreadExecutionState
            self.set_state.argtypes=[ctypes.c_uint32];self.set_state.restype=ctypes.c_uint32
            if not self.set_state(0x80000001):raise OSError('Temporary SYSTEM_REQUIRED request failed')
            self.active=True
        return self
    def __exit__(self,exc_type,exc_value,traceback):
        if self.active:
            self.set_state(0x80000000);self.active=False
        return False
