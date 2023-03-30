#!/bin/bash

# python3 fix_task.py 
# python3 fix_task.py --model-take-db True --model-take-abs False --model-take-amp True
# python3 fix_task.py --model-take-db True --model-take-abs True --model-take-amp True
# python3 fix_task.py --model-depth 4 
# python3 fix_task.py --model-depth 4 --model-take-db True --model-take-abs False --model-take-amp True
# python3 fix_task.py --model-depth 4 --model-take-db True --model-take-abs True --model-take-amp True
# python3 fix_task.py --model-depth 6 
# python3 fix_task.py --model-depth 6 --model-take-db True --model-take-abs False --model-take-amp True
# python3 fix_task.py --model-depth 6 --model-take-db True --model-take-abs True --model-take-amp True

# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 8 --model-s4-hidden-size 8
# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 8 --model-s4-hidden-size 16
# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 16 --model-s4-hidden-size 8
# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 8 --model-s4-hidden-size 32
# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 32 --model-s4-hidden-size 8
# python3 fix_task.py --model-depth 4 --model-inner-audio-channel 32 --model-s4-hidden-size 32

python3 fix_task.py --model-depth 4 --model-inner-audio-channel 32 --model-s4-hidden-size 16
python3 fix_task.py --model-depth 4 --model-inner-audio-channel 16 --model-s4-hidden-size 32
