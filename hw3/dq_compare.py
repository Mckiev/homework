import subprocess

subprocess.run('python3 run_dqn_atari_sweep.py --model xsmall --doubleQ' shell = True)
subprocess.run('python3 run_dqn_atari_sweep.py --model xsmall' shell = True)
