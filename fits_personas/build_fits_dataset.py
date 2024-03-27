import subprocess
import re
command = ['parlai', 'display_data', '-t', 'fits']
output = subprocess.run(command, capture_output=True, text=True).stdout

# Search for the datapath using regular expressions
datapath = re.search(r'datapath: (.+)', output).group(1)

# Move dataset to the location of this script
subprocess.run(['mv', datapath+"/fits", 'fits_personas'])