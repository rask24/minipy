from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def build_extensions(self):
        pass


setup(
    name="minipy",
    version="0.1.0",
    author="Your Name",
    description="A minimal numpy-like library with CUDA support",
    packages=["minipy"],
    package_dir={"": "src"},
    ext_modules=[
        Extension(
            "minipy.backend.cuda_ops",
            ["src/minipy/backend/cuda_ops.cu"],
        )
    ],
    cmdclass={"build_ext": CustomBuildExt},
    python_requires=">=3.12",
)
