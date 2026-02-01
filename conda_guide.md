# Conda Guide

### To create a conda env from a file (from git repo)
```bash
conda env create -f conda_env.yml
conda activate gaitkeeper
```

### To write your conda env to a file (to push to git repo)
```bash
conda env export --from-history > conda_env.yml
# Remove the "prefix" line at the bottom of the file with your OS absolute path.
```
