# Rulekit
 
This package is python wrapper for [RuleKit](https://github.com/adaa-polsl/RuleKit) library - a versatile tool for rule learning. 
 
Based on a sequential covering induction algorithm, it is suitable for classification, regression, and survival problems.
 
## Installation
 
> **NOTE**: 
This package is a wrapper for Java library, it requires Java (version 1.8.0 tested) to be installed on the computer. Both Open JDK and Oracle implementations are supported.
## 
 
```bash
pip install rulekit
 
# after installation
python -m rulekit download_jar
```
 
The second command will fetch the latest RuleKit library jar file from its Github releases page. It is required to use this package.
 
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
from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from sklearn.datasets import load_iris
 
X, y = load_iris(return_X_y=True)
 
clf = RuleClassifier()
clf.fit(X, y)
prediction = clf.predict(X)
 
print(prediction)
```
 
## Documentation
 
Full documentation is available [here](https://adaa-polsl.github.io/RuleKit-python/)

