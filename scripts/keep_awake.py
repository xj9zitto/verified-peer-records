"""keep_awake.py <pid-file> -- hold off Windows sleep while a campaign runs.

Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and
keeps it set until <pid-file> disappears.  Changes no power settings;
the request is released when this process exits.
"""
import ctypes
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
flag = Path(sys.argv[1])
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
while flag.exists():
    time.sleep(30)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
