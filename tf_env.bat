@echo off
@REM Tensorflow and keras
conda remove -y -n tf_env --all
conda create -y -n tf_env python=3.7
conda activate tf_env
python -m pip install -U tensorflow-gpu==2.4.1 tensorflow_datasets
python -m pip install -U keras==2.4.3
python -m pip install -U pillow matplotlib flake8 autopep8

@REM GA Library
python -m pip install -U deap
