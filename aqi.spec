# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: build a single-file ``aqi`` executable from src/cli.py."""
import os
import pathlib

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = pathlib.Path(SPEC).resolve().parent

hiddenimports = [
    "torch", "torch.nn", "torch.nn.functional", "torch.distributions",
    "sklearn", "sklearn.preprocessing", "sklearn.model_selection",
    "sklearn.metrics", "scipy", "scipy.sparse", "scipy.spatial", "scipy.stats",
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "plotly", "plotly.graph_objects", "plotly.io", "plotly.validators",
    "folium", "branca", "geopandas", "fiona", "pyproj", "shapely",
    "shapely.geometry", "shapely.ops", "pyogrio",
]
hiddenimports += collect_submodules("imblearn")

datas = []
for pkg in ("plotly", "folium", "branca", "geopandas", "pyproj", "fiona"):
    datas += collect_data_files(pkg)
datas += collect_data_files("imblearn")

# pyogrio: PEP-384 abi3 extensions + bundled GDAL lib are not picked up by
# static analysis, so ship the whole package dir + its .libs sibling as data.
import importlib as _il
_pyogrio = _il.import_module("pyogrio")
_pydir = os.path.dirname(_pyogrio.__file__)
_pylibs = os.path.join(os.path.dirname(_pydir), "pyogrio.libs")
datas += [(_pydir, "pyogrio")]
if os.path.isdir(_pylibs):
    datas += [(_pylibs, "pyogrio.libs")]
del _pyogrio, _pydir, _pylibs
# project data + standalone scripts invoked via runpy
datas += [
    (str(ROOT / "data" / "external"), "data/external"),
    (str(ROOT / "data" / "processed"), "data/processed"),
    (str(ROOT / "data" / "raw" / "airquality_uci.csv"), "data/raw"),
    (str(ROOT / "data" / "raw" / "india_city_day.csv"), "data/raw"),
    (str(ROOT / "data" / "raw" / "delhi_cpcb_2024_25.csv"), "data/raw"),
    (str(ROOT / "data" / "raw" / "india_cities.csv"), "data/raw"),
    (str(ROOT / "data" / "raw" / "openaq"), "data/raw/openaq/"),
    (str(ROOT / "scripts" / "benchmark_models.py"), "scripts"),
    (str(ROOT / "benchmarking" / "run_sweep.py"), "benchmarking"),
]

a = Analysis(
    ["src/cli.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["kaleido", "matplotlib.tests", "tkinter", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="aqi",
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