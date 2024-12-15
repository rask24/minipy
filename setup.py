import os
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def build_extensions(self):
        cuda_ext = None
        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                cuda_ext = ext
                break

        if cuda_ext:
            cuda_sources = [s for s in cuda_ext.sources if s.endswith(".cu")]
            obj_dir = os.path.dirname(self.get_ext_fullpath(cuda_ext.name))

            for cuda_source in cuda_sources:
                obj = os.path.join(
                    obj_dir, os.path.splitext(os.path.basename(cuda_source))[0] + ".o"
                )
                os.makedirs(obj_dir, exist_ok=True)

                nvcc_cmd = [
                    "nvcc",
                    "-c",
                    cuda_source,
                    "-o",
                    obj,
                    "-std=c++11",
                    "-O2",
                    "-Xcompiler",
                    "-fPIC",
                    "--ptxas-options=-v",
                    "-arch=sm_80",
                ]
                for include_dir in cuda_ext.include_dirs:
                    nvcc_cmd.extend(["-I", include_dir])

                self.spawn(nvcc_cmd)
                if obj not in self.compiler.objects:
                    self.compiler.objects.append(obj)

            cuda_ext.sources = [s for s in cuda_ext.sources if not s.endswith(".cu")]

        super().build_extensions()


cuda_home = "/usr/local/cuda-12.6"

ext_modules = [
    Extension(
        name="minipy.backend.array_ops",
        sources=[
            "src/minipy/backend/array_ops.cpp",
            "src/minipy/backend/array_ops.cu",
            "src/minipy/backend/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            os.path.join(cuda_home, "include"),
            os.path.dirname(os.path.abspath(__file__)) + "/src/minipy/backend",
        ],
        library_dirs=[os.path.join(cuda_home, "lib64")],
        libraries=["cudart"],
        extra_compile_args=["-std=c++11"],
        runtime_library_dirs=[os.path.join(cuda_home, "lib64")],
        language="c++",
    )
]

setup(
    name="minipy",
    packages=["minipy"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
