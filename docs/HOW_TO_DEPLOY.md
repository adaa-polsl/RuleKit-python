# Deployment instruction

> Don't forger to change version before deployment!
> Deployment **MUST** be done from the `main` branch only!

### 1. Create tag in Github repository 
Create tag on current commit named `v{CURRENT_VERSION}`

### 2. Create release in Github
Use previously created tag for it.

> Documentation will be automatically generated on new version release 

### 3. Deploy to pypi
In root repository directory:
```
rm -r ./dist
python -m build
python -m twine check dist/*
python -m twine upload  dist/*
```
> For the last command use `__token__` as username and your token value as password when prompted.