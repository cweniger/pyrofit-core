from setuptools import setup, find_packages

setup(
    name='gpyro',
    description="pyro.ai-based modeling and inference for research at the GRAPPA institute",
    version='0.1',
    packages=['gpyro'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        gpyro=gpyro.main:cli
    ''',
)
