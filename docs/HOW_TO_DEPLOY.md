# Deployment instruction

> Deployment **MUST** be done from the `main` branch only!

## 1. Change version

Version is stored in few places (unfortunately):
* `main.py` file, `__RULEKIT_RELEASE_VERSION__` variable should be set to the *.jar version of the RuleKit library used by this specific package version
* `main.py` file, `__VERSION__` variable specify additional version number which should be elevated only when given release includes changes in Python but still uses the same *.jar version as the last existing release. 
* `setup.py` version should be the same as `__VERSION__` in `main.py` file

To sum up:
* If you migrate to new *.jar file -> change `__RULEKIT_RELEASE_VERSION__` and version in `setup.py`
* If you change something in the package itself and continue to use the same version of RuleKit jar file -> bump only the last number of `__VERSION__`

> ⚠️ **Always update version in `setup.py` to match `__VERSION__` from the `main.py` file!**

### 2. Create tag in Github repository 
Create tag on current commit named `v{CURRENT_VERSION}`

### 2. Create release in Github
Use previously created tag for it.

> Documentation will be automatically generated on new version release 

### 3. Deploy to pypi
In root repository directory:
```
rm -r ./dist
python setup.py sdist
python -m twine check ./dist/*
python -m twine upload  ./dist/*
```
> For the last command use `__token__` as username and your token value as password when prompted.