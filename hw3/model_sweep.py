import subprocess

for model in ['s1', 's2', 's3','b', 'm', 'def256', 'def128', 'def']:
    subprocess.run('python3 run_dqn_atari.py --model %s --doubleQ --subdir ModelSweep'%model, shell = True)

subprocess.run('python3 plot.py Logz/ModelSweep --pic_name model_sweep.png', shell = True)