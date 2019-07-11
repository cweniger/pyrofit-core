from setuptools import setup

setup(
    name='pyrofit_core',
    description="pyro.ai-based modeling and inference for astroparticle physics research",
    version='0.1',
    packages=['pyrofit.core'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        pyrofit=pyrofit.core.main:cli
    ''',
    zip_safe=False,
)
