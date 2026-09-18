# -*- mode: python ; coding: utf-8 -*-
"""Build the one-directory ``betterborg`` bundle with all package assets.

Shared libraries keep stable paths inside the bundle. A one-file executable
unpacks them to a new temporary directory on every launch, and macOS verifies
each newly written library again, which made startup take tens of seconds.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("betterborg_cli")

analysis = Analysis(
    ["src/betterborg_cli/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    # The shared reporter is not wired to a workflow until later progress-control
    # tasks, but it and Rich must already be viable in this first shipping change.
    hiddenimports=["betterborg_cli.progress"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="betterborg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="betterborg",
)
