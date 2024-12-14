# setup.py
from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        name="minipy.backend.array_ops",
        sources=["src/minipy/backend/array_ops.cpp", "src/minipy/backend/bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            os.path.dirname(os.path.abspath(__file__)) + "/src/minipy/backend",
        ],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="minipy",
    packages=["minipy"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
