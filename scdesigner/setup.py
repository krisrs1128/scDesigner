from setuptools import setup, find_packages

setup(
    name='scdesigner',
    version='0.0.999',
    packages=find_packages(),
    install_requires=['torch', 'numpy'],
    description='Interactive simulation for rigorous and transparent multi-omics analysis',
    author='Kris Sankaran',
    author_email='ksankaran@wisc.edu',
    url='https://github.com/krisrs1128/scDesigner',
)
