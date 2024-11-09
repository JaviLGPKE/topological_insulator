from setuptools import setup, Command
import subprocess
import os


class CMakeBuild(Command):
    description = "Builds C++ code using CMakeLists.txt"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        source_directory = os.path.join(os.getcwd(), 'src', 'solver_dolfinx', 'cpp')
        build_directory = os.path.join(source_directory, 'build')
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)
        # Run cmake to configure the build
        subprocess.check_call(['cmake', '-S', source_directory, '-B', build_directory])
        # Run cmake to build
        subprocess.check_call(['cmake', '--build', build_directory])


setup(
    cmdclass={
        # 'build_ext': CMakeBuild,
    })
