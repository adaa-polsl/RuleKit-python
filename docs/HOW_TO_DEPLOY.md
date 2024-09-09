# Deployment instruction

> Don't forger to change version before deployment!
> Deployment **MUST** be done from the `main` branch only!

### 1. Update documentation on Github Pages
> Make sure you installed docs dependencies in **SEPARATE** virtual env and activated it

In `docs` directory call call:

```bash
build.py <VERSION_NUMBER>
```
e.g. 
```bash
build.py 2.18.0.0
```

### 2. Update badges

In repo root directory:

1. Update coverage badge
```bash
python -m coverage run -m unittest discover ./tests 
python -m coverage xml -o ./docs/reports/coverage/coverage.xml  
python -m coverage html -d ./docs/reports/coverage/
genbadge coverage -i ./docs/reports/coverage/coverage.xml  -o ./docs/badges/coverage-badge.svg
```

2. Update test badge

```bash
rm -r ./docs/reports/junit
mkdir ./docs/reports/junit
python -m junitxml.main --o ./docs/reports/junit/junit.xml
python -m junit2htmlreport ./docs/reports/junit/junit.xml ./docs/reports/junit/report.html
genbadge tests -i ./docs/reports/junit/junit.xml -o ./docs/badges/test-badge.svg
```

3. Update flake8 badge

```bash
flake8 ./rulekit --exit-zero --format=html --htmldir ./docs/reports/flake8 --statistics --tee --output-file ./docs/reports/flake8/flake8stats.txt
genbadge flake8 -i ./docs/reports/flake8/flake8stats.txt -o ./docs/badges/flake8-badge.svg
```

### 3. Create tag in Github repository 
Create tag on current commit named `v{CURRENT_VERSION}`

### 4. Create deployment in Github
Use previously created tag for it.

### 5. Deploy to pypi
In root repository directory:
```
rm -r ./dist
python -m build
python -m twine check dist/*
python -m twine upload  dist/*
```
> For the last command use `__token__` as username and your token value as password when prompted.