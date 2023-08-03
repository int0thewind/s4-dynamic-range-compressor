#!/bin/zsh

pipenv run python3 -m s4drc fit --trainer.accelerator cpu --trainer.default_root_dir ./experiment-result --trainer.max_epochs 100 --trainer.logger WandbLogger --trainer.logger.init_args.project S4DRC --trainer.logger.init_args.log_model all
