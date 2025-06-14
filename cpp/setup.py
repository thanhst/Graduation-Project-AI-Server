import os
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j4']

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(['cmake', os.path.abspath('.')] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

        print(f"\n[INFO] Extension built at: {extdir}")

setup(
    name='rtpimageproc',
    version='0.1.0',
    author='thanhxk2003',
    author_email='thanhxk2003@gmail.com',
    description='Process RTP image and run ONNX model',
    ext_modules=[CMakeExtension('rtpimageproc')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
