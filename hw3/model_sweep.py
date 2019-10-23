import subprocess

for model in ['1c', 'xxsmall', 'xsmall', 'small', 'def']:
    subprocess.run('python3 run_dqn_atari_sweep.py --model %s --doubleQ'%model, shell = True)
