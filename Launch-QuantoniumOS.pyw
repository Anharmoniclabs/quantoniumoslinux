from pathlib import Path
import os, subprocess, sys
base=Path(__file__).resolve().parent
env=os.environ.copy();env['QUANTONIUM_HOME']=str(base)
python=Path(sys.executable).with_name('python.exe')
env['QUANTONIUM_PYTHON']=str(python)
native=base/'runtime/desktop-shell/src-tauri/target/release/quantoniumos-desktop.exe'
if native.exists():
    subprocess.Popen([str(native)],cwd=base/'runtime',env=env)
else:
    import ctypes
    ctypes.windll.user32.MessageBoxW(0,f'The QuantoniumOS desktop app has not been built yet.\n\nRun Build-Desktop.ps1, then launch again.\n\nExpected: {native}','QuantoniumOS',0x10)
