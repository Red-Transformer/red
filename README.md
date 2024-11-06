# Red Transformer - _Red Teaming agentic systems_

## Install

### With Virtualenv

If you already have the right python version installed on your computer, you can initiate the project with:

```shell
python -m venv .venv 
source .venv/bin/activate
pip install poetry 
poetry install
```

> You can check the python version using `python --version`

### With Conda

To get the correct Python version, you can use miniconda. First, get it on the [download page](https://docs.anaconda.com/miniconda/#miniconda-latest-installer-links). On linux you can use:

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Then source it and create a env with the proper version:

```shell
source ~/miniconda3/bin/activate
conda create -n redvenv python=3.12
conda activate redenv

# And then you can build a virtual env
python -m venv .venv 

conda deactivate  # get out of the conda redenv
conda deactivate  # get out of the conda base env 
source .venv/bin/activate  # get in the virtual env that has python 3.12

pip install poetry 
poetry install
```

## VSCode configuration

Enable Ruff as formater and MyPy as type checker:

- Install the Ruff extension
- Install the Mypy Type Checker extension

```json
"editor.defaultFormatter": "charliermarsh.ruff",
"editor.formatOnSave": true,
"notebook.formatOnSave.enabled": true,
"editor.codeActionsOnSave": {
    "source.fixAll.ruff": "always",
    "source.organizeImports.ruff": "always",
    "source.fixAll": "always"
},
"ruff.exclude": [],
"ruff.lint.enable": true,
"notebook.defaultFormatter": "charliermarsh.ruff",
```

Configure Jupyter to use the root of the project as working directory so that the local modules can be called properly from anywhere.

```json
"jupyter.notebookFileRoot": "${workspaceFolder}"
```
