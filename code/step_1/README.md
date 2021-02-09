# Step 1: Ensemble RL

# Files

`common`: Python helpers, modules, packages.

`input-data`: The data needed for training. e.g. configurations.

`output-data`: The data produced by training. e.g. learning curves.

`.gitignore`: A file that contains files that don't need to be synchronized to GitHub or VACC.

`0.init.py`: Initialize a new experiment environment.

`submit.sh`: A script for VACC sbatch job submitting.

`to_vacc.sh`: Synchronize this experiment to VACC.

`train.py`: The core training script. Using different parameters to control different behaviors of the training algorithm.
