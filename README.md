[![Coverage Status](https://adaa-polsl.github.io/RuleKit-python/badges/coverage-badge.svg?dummy=8484744)](https://adaa-polsl.github.io/RuleKit-python/reports/coverage/index.html)
[![Tests Status](https://adaa-polsl.github.io/RuleKit-python/badges/test-badge.svg?dummy=8484744)](https://adaa-polsl.github.io/RuleKit-python/reports/junit/report.html)
[![Flake8 Status](https://adaa-polsl.github.io/RuleKit-python/badges/flake8-badge.svg?dummy=8484744)](https://adaa-polsl.github.io/RuleKit-python/reports/flake8/index.html)
[![PyPI](https://img.shields.io/pypi/v/rulekit?label=pypi%20package)](https://pypi.org/project/rulekit/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rulekit)](https://pypi.org/project/rulekit/)

# Rulekit


This package is python wrapper for [RuleKit](https://github.com/adaa-polsl/RuleKit) library - a versatile tool for rule learning. 
 
Based on a sequential covering induction algorithm, it is suitable for classification, regression, and survival problems.
 
## Installation
 
> **NOTE**: 
This package is a wrapper for Java library, and requires Java Development Kit version 8 or later to be installed on the computer. Both Open JDK and Oracle implementations are supported.
## 
 
```bash
pip install rulekit
```
 
## Running tests
 
If you're running tests for the first time (or you want to update existing tests resources) you need to download tests resources from RuleKit repository. You can do it by running:
```
python tests/resources.py download
```
Runing tests:    
In directory where `setup.py` file exists.
```
python -m unittest discover ./tests
```
 
## Sample usage
 
```python
from sklearn.datasets import load_iris

from rulekit.classification import RuleClassifier
 
X, y = load_iris(return_X_y=True)
 
clf = RuleClassifier()
clf.fit(X, y)
prediction = clf.predict(X)
 
print(prediction)
```
 
## Documentation
 
Full documentation is available [here](https://adaa-polsl.github.io/RuleKit-python/)

## Licensing

The software is publicly available under [GNU AGPL-3.0](https://github.com/adaa-polsl/RuleKit-python/blob/main/LICENSE) license. Any derivative work obtained under this license must be licensed under the AGPL if this derivative work is distributed to a third party. For commercial projects that require the ability to distribute RuleKit code as part of a program that cannot be distributed under the AGPL, it may be possible to obtain an appropriate license from the authors.
