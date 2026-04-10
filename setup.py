import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.sep):
            extdir += os.sep

        cfg = "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = ["--config", cfg, "-j"]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )


setup(
    name="carcassonne-engine",
    version="0.1.0",
    description="Carcassonne game engine with C++ backend",
    packages=find_packages(),
    ext_modules=[CMakeExtension("_carcassonne_engine", sourcedir="carcassonne_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
