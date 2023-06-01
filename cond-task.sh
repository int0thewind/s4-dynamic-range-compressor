#!/bin/bash

pipenv run python3 cond_task.py
pipenv run python3 cond_task.py --model-take-side-chain True
pipenv run python3 cond_task.py --model-take-tanh True
pipenv run python3 cond_task.py --model-take-tanh True --model-take-side-chain True
