import os
import sys
from setuptools import setup, find_packages

print("Installing Dexterous Arm Hardware models package!.")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='dexterous_arm_models',
    version='1.0.0',
    packages=find_packages(),
    description='Models that can be deployed on the dexterous arm.',
    long_description=read('README.md'),
    url='https://github.com/NYU-robot-learning/Dexterous-Arm-Models',
    author='Sridhar Pandian',
)