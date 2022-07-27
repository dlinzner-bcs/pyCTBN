from setuptools import setup

setup(
    name='pyctbnlearn',
    version='0.1.0',
    description='A package for sampling and learning ctbns from data',
    url='https://github.com/dlinzner-bcs/pyCTBN',

    packages=['ctbn'],
    install_requires=['scipy',
                      'numpy',
                      'plotly'
                      ],
    python_requires=">=3.6",
)
