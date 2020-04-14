import sys
import os

def append_working_directory():
    current_directory = os.getcwd()
    index = current_directory.find('/', 1, len(current_directory))
    dirlist = os.listdir(current_directory[:index])
    if 'chakrm' in dirlist:
        ssh_script_location = '/chakrm/workspace/2019-MastersProject/'
        sys.path.append(os.path.abspath('..')+ssh_script_location)
    else:
        sys.path.append(os.path.abspath('..'))