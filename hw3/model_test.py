import subprocess
import time

start_time = time.time()
subprocess.run('python3 run_dqn_atari_fc.py', shell = True)
fc_time = time.time() - start_time
start_time = time.time()
subprocess.run('python3 run_dqn_atari_vsmall.py', shell = True)
vsmall_time = time.time() - start_time

with open('res/times.txt', 'w') as f:
        f.write('FC time = ' + str(fc_time) + '\n' + 'vsmall time = ' + str(vsmall_time))


