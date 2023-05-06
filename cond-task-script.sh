#!/bin/bash

pipenv run python3 cond_task.py
pipenv run python3 cond_task.py --model-film-take-batchnorm True
pipenv run python3 cond_task.py --model-activation PReLU
