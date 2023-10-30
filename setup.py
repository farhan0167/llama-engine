from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='llama-engine',
    version='1.0',
    packages=find_packages(),
    install_requires=install_requires,
)
