#preparingdata.py
#!/usr/bin/env python3
import os
import sys
import string
#from fileread import *

filetoread= 'log_tcp_complete_exp8.txt'
filetowrite= 'tcp_exp8.csv'
with open(filetoread) as f:
	tcpData=f.read().replace(' ', ',')

with open(filetowrite, "w") as f:
	f.write(tcpData)
