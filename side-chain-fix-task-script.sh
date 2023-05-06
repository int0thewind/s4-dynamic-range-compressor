#!/bin/bash

# We already know that GELU is the best non-linear activation function,
# that the model version 4 is the best model version,
# that decibel conversion is not necessary,
# that the best loss function is ESR+DC+Multi-STFT,
# and that the best model depth is 4.
# We just need to run the test again for the sake of completeness.
pipenv run python3 fix_task.py

# Remove residual connection
pipenv run python3 fix_task.py --model-take-side-chain False

# Test model version
pipenv run python3 fix_task.py --model-version 0
pipenv run python3 fix_task.py --model-version 1
pipenv run python3 fix_task.py --model-version 2
pipenv run python3 fix_task.py --model-version 3

# Test activation layer
pipenv run python3 fix_task.py --model-activation tanh
pipenv run python3 fix_task.py --model-activation sigmoid
pipenv run python3 fix_task.py --model-activation ReLU
pipenv run python3 fix_task.py --model-activation PReLU

# Test decibel conversion
pipenv run python3 fix_task.py --model-convert-to-decibels True

# Test loss function
pipenv run python3 fix_task.py --loss MAE
pipenv run python3 fix_task.py --loss MSE
pipenv run python3 fix_task.py --loss ESR+DC
pipenv run python3 fix_task.py --loss Multi-STFT
pipenv run python3 fix_task.py --loss MAE+Multi-STFT

# Test model depth
pipenv run python3 fix_task.py --model-depth 2
pipenv run python3 fix_task.py --model-depth 6

# But finally, we still need to run the hidden size test.
# The default audio channel number is 16 and s4 hidden size is 16.
pipenv run python3 fix_task.py --model-inner-audio-channel 8 --model-s4-hidden-size 4
pipenv run python3 fix_task.py --model-inner-audio-channel 8 --model-s4-hidden-size 8
pipenv run python3 fix_task.py --model-inner-audio-channel 8 --model-s4-hidden-size 16
pipenv run python3 fix_task.py --model-inner-audio-channel 16 --model-s4-hidden-size 4
pipenv run python3 fix_task.py --model-inner-audio-channel 16 --model-s4-hidden-size 8
pipenv run python3 fix_task.py --model-inner-audio-channel 16 --model-s4-hidden-size 16
pipenv run python3 fix_task.py --model-inner-audio-channel 16 --model-s4-hidden-size 32
pipenv run python3 fix_task.py --model-inner-audio-channel 32 --model-s4-hidden-size 4
pipenv run python3 fix_task.py --model-inner-audio-channel 32 --model-s4-hidden-size 8
pipenv run python3 fix_task.py --model-inner-audio-channel 32 --model-s4-hidden-size 16
pipenv run python3 fix_task.py --model-inner-audio-channel 32 --model-s4-hidden-size 32
