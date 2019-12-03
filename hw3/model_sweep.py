import subprocess

for model in ['s1', 's2', 's3', 'def256', 'def128', 'def']:
    subprocess.run('python3 run_dqn_atari.py --model %s --doubleQ'%model, shell = True)
