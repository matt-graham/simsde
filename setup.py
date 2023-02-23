#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='simsde',
    version='0.1.0',
    description='Numerical integrators for simulating stochastic differential equations',
    author='Matt Graham and Yuga Iguchi',
    url='https://github.com/matt-graham/sdesim.git',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'symnum==0.1.2',
        'sympy>=1.10',
    ],
)
