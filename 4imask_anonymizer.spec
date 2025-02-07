# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('res', 'res')]
binaries = []
hiddenimports = ['ultralytics', 'skimage._shared.geometry']
datas += collect_data_files('openvino')
binaries += collect_dynamic_libs('openvino')
hiddenimports += collect_submodules('openvino')

a = Analysis(
    ['4imask_anonymizer.py'],
    pathex=['/D:/OneDrive - 4itec/Documents/GitHub/4imask'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyi_splash = Splash(
    'res/icons/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    custom_text="Loading...",
    full_screen=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyi_splash,
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='4imask_anonymizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='res/icons/4itec.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='4imask_anonymizer',
)
