import subprocess
import time


start_time = time.time()
subprocess.run('python3 run_dqn_atari_fc.py', shell = True)
fc_time = time.time() - start_time

start_time = time.time()
subprocess.run('python3 run_dqn_atari.py', shell = True)
def_time = time.time() - start_time

start_time = time.time()
subprocess.run('python3 run_dqn_atari_xsmall.py', shell = True)
xsmall_time = time.time() - start_time

with open('res/times1.txt', 'w') as f:
        f.write('def time = ' + str(def_time) + '\n' + 'xsmall time = ' + str(xsmall_time) + '\n' + 'fc time = ' + str(fc_time))


