import sys
import os

def get_working_directory():
    current_directory = os.getcwd()
    index = current_directory.find('/', 1, len(current_directory))
    dirlist = os.listdir(current_directory[:index])
    if 'chakrm' in dirlist:
        return '/home/chakrm/workspace/2019-MastersProject/source/helpers/'
    return os.path.dirname(os.path.realpath(__file__))
