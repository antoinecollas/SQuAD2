from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()
setup(
    name='transformer',
    version='0.1',
    description='Transformer implementation',
    license='MIT',
    long_description=long_description,
    author='Antoine Collas',
    packages=find_packages(),
)