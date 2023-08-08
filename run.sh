#!/bin/zsh

# pipenv run python3 -m s4drc fit \
# --model.inner_audio_channel 32 \
# --model.s4_hidden_size 4 \
# --model.depth 4 \
# --model.final_tanh tanh \
# --model.take_side_chain False \
# --model.side_chain_tanh False \
# \
# --trainer.default_root_dir ./experiment-result/run-01/ \
# --trainer.max_epochs 60 \
# \
# --trainer.logger WandbLogger \
# --trainer.logger.init_args.project S4DRC \
# --trainer.logger.init_args.log_model all \
# \
# --trainer.callbacks+ ModelCheckpoint \
# --trainer.callbacks.init_args.dirpath ./experiment-result/run-01/ \
# --trainer.callbacks.init_args.save_top_k 5 \
# --trainer.callbacks.init_args.monitor 'Validation Loss' \
# --trainer.callbacks.init_args.save_last True \
# # \
# # --trainer.callbacks+ SaveConfigCallback \
# # --trainer.callbacks.init_args.config_filename ./experiment-result/run-01/config.yaml

for f in experiment-result/*
do
pipenv run python3 -m s4drc fit $f/config.yaml
done
# pipenv run python3 -m s4drc fit -c 
