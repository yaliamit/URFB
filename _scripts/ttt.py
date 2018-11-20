import subprocess as commands
import time
import os
import sys

try:
    commands.check_output('ssh yaliamit@midway2.rcc.uchicago.edu "cd /home/yaliamit/Desktop/Dropbox/Python; git pull" ', shell=True)
except:
    print('pull failed')