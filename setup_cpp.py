"""Build the C++ extension in place:  python setup_cpp.py build_ext --inplace

Source: src/sensorsim/cpp/fix_defects.cpp
Output: src/sensorsim/_fixcore*.pyd, importable as sensorsim._fixcore.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "sensorsim._fixcore",
    ["src/sensorsim/cpp/fix_defects.cpp"],
    cxx_std=17,
)

setup(
    name="sensorsim-fixcore",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
)