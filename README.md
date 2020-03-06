WARNING: This is research software under development. Use at your own risk.


# pyrofit_core
Data analysis build around pyro.ai

## Documentation

Documentation can be found here: https://pyrofit-core.readthedocs.io

## Installation

Installation happens as usual via
```
python3 setup.py install
```
This will install some new command line tool `pyrofit`, which should be accessible and in `PATH` immediately after installation.
Please run
```
pyrofit --help
```
to see the viable commands.

## Examples

Basic examples can be found in `pyrofit_core/pyrofit/examples`. All relevant python code is in `minimal.py`. The examples can be run
for example with
```
pyrofit linear.py fit
```
By default, a maximum a-posteriori (MAP) fit is performed, using Delta-guides. Inspect the `*.yaml` files for more details.

Afterwards 
```
pyrofit linear.py ppd ppd.npz
```
would draw from the posterior predictive distribution (by default just from the MAP).
