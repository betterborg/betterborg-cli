# -*- mode: python ; coding: utf-8 -*-
"""Build the local one-file ``borg`` executable with all package assets."""

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
    analysis.binaries,
    analysis.datas,
    [],
    name="borg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
