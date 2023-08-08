#!/bin/zsh

for f in ./experiment-result/**/config.yaml
do
pipenv run python3 -m s4drc fit -c $f
done
