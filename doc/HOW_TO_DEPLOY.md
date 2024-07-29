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

In `doc` directory call call:
```bash
make.bat`
```
1. Rename directory `/doc/build/html` to `/doc/build/v{CURRENT_VERSION}`.

2. Change branch to `docs` and copy `/doc/build/v{CURRENT_VERSION}` directory to the root repo directory.

3. Go to `index.html` file in the root repo directory and add modify versions list manually:
```html
    <li class="toctree-l1">
    <!--Remove "(latest)" from the previous version item and add it to the currently latest version item -->
    <a class="reference internal" href="v1.7.6/index.html"
        >v.1.7.6</a 

    >
    </li>
    <li class="toctree-l1">
    <!-- Update href to match your folder path -->
    <a class="reference internal" href="v1.7.14.0/index.html"
        >v.1.7.14.0 (latest)</a
    >
    </li>
```
