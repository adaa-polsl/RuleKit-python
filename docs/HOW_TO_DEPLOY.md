# Deployment instruction

> Don't forger to change version before deployment!
> Deployment **MUST** be done from the `main` branch only!

### 1. Create tag in Github repository 
Create tag on current commit named `v{CURRENT_VERSION}`

### 2. Create deployment in Github
Use previously created tag for it.

### 1. Deploy to pypi
In root repository directory:
```
rm -r ./dist
python -m build
python -m twine check dist/*
python -m twine upload  dist/*
```
> For the last command use `__token__` as username and your token value as password when prompted.

### 2. Update documentation on Github Pages
> Make sure you installed docs dependencies in **SEPARATE** virtual env and activated it

In `docs` directory call call:

```bash
build.py <VERSION_NUMBER>
```
e.g. 
```bash
build.py 2.18.0.0
```

### 3. Update badges

In repo root directory:

1. Update coverage badge
```bash
python -m coverage run -m unittest discover ./tests 
python -m coverage xml -o ./reports/coverage/coverage.xml  
python -m coverage html -d ./reports/coverage/
genbadge coverage -i ./reports/coverage/coverage.xml  -o ./badges/coverage-badge.svg
```

2. Update test badge

```bash
mkdir ./reports/junit
python -m junitxml.main --o ./reports/junit/junit.xml

genbadge tests -o ./badges/test-badge.svg
```

3. Update flake8 badge

```bash
flake8 ./rulekit --exit-zero --format=html --htmldir ./reports/flake8 --statistics --tee --output-file ./reports/flake8/flake8stats.txt
genbadge flake -o ./badges/flake8-badge.svg
```