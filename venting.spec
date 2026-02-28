# venting.spec
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

hidden_imports = [
    'scipy', 'scipy.integrate', 'scipy.integrate._ivp',
    'scipy.integrate._ivp.radau', 'scipy.integrate._ivp.common',
    'scipy.integrate._ivp.base', 'scipy.linalg',
    'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack',
    'scipy.sparse', 'scipy.sparse.linalg', 'scipy.special',
    'scipy.special._ufuncs', 'scipy._lib.messagestream',
    'numpy', 'numpy.core._multiarray_umath',
    'PySide6', 'PySide6.QtCore', 'PySide6.QtGui',
    'PySide6.QtWidgets', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
    'pyqtgraph', 'pyqtgraph.graphicsItems', 'pyqtgraph.widgets',
    *collect_submodules('venting'),
]

datas = [
    *collect_data_files('pyqtgraph'),
    *collect_data_files('scipy'),
]

a = Analysis(
    ['src/venting/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'setuptools'],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='venting',
    debug=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll',
        'Qt6OpenGL.dll', 'Qt6OpenGLWidgets.dll', 'PySide6/*.pyd',
    ],
    console=False,
    runtime_tmpdir=None,
)
