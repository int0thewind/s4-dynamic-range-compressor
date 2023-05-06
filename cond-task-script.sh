#!/bin/bash

# I should test:
# 1. res connection or not
# 2. batchnorm or not
# 3. tanh or not

pipenv run python3 cond_task.py
pipenv run python3 cond_task.py --model-take
pipenv run python3 cond_task.py --model-film-take-batchnorm True
pipenv run python3 cond_task.py --model-activation PReLU
