# -*- coding: utf-8 -*-
import sys
from io import StringIO, BytesIO
import glob
import os
import stat
import time
from tqdm import tqdm
from zipfile import ZipFile
import shutil

def create_folder(destination_path):
    if not os.path.exists(destination_path):
        current_path = os.path.dirname(os.path.realpath(__file__))
        os.chmod(current_path + "/00_unzip_patents.py", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        os.makedirs(destination_path)

# this works for every situation, but you should define the folder_level
def add_folder(string, folder_level):
    if folder_level > len(string.split('/')):
        return string + 'unzipped/'
    return th.'/'.join(s for s in string.split('/')[0:folder_level]) + '/unzipped/' + '/'.join(s for s in string.split('/')[folder_level:])
    return '/'.join(s for s in string.split('/')[0:folder_level]) + '/unzipped/' + '/'.join(s for s in string.split('/')[folder_level:])

# # this works for USI server environment!
# def add_folder(string, folder_level):
#     return '/'.join(s for s in string.split('/')[0:folder_level+2]) + '/unzipped/' + string.split('/')[-1] + '/'

# def get_destination_path(source_path, folder_level):
#     destination_path = []
#     for path in source_path:
#         index = path.find('/')
#         if index != -1:
#             path = add_folder(path, folder_level)
#             destination_path.append(path)
#             create_folder(path)
#         else:
#             destination_path.append('unzipped/')
#     return destination_path

def get_destination_path(source_path, folder_level):
    destination_path = []
    for path in source_path:
        path = add_folder(path, folder_level)
        destination_path.append(path)
        create_folder(path)
    return destination_path

def get_list_files(path, extension):
    return glob.glob(path + '/*.' + extension)

def unzip_patents(source_path, destination_path):
    """ unzip_patents """
    for index, path in enumerate(source_path):
        for path_filename in get_list_files(path, 'zip'):
            try:
                print("file: ", path_filename)

                zip_ref = ZipFile(path_filename, 'r')
                zip_ref.extractall(destination_path[index])
                zip_ref.close()
            except:
                print("WARNING!!!! error - check out the patent: ", path_filename)
                continue

def move_patents(source_path, destination_path):
    """ unzip_patents """
    for index, path in enumerate(source_path):
        files = get_list_files(path, 'xml')
        for i in tqdm(range(len(files))):
            try:
                path_filename = files[i]
                filename = path_filename[path_filename.rfind('/')+1:]
                # print("file: ", path_filename)

                shutil.copyfile(path_filename, destination_path[index] + filename)
            except:
                print("WARNING!!!! Check out the patent: ", path_filename)
                continue

if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            source_path = sys.argv[1]
            folder_level = sys.argv[2]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path = source_path + '/*'
            source_path = glob.glob(source_path)
            source_path = [path + '/' for path in source_path]
            print("source path: %s" % source_path)
            folder_level = int(folder_level)
            print("folder destination level: %s" % folder_level)
        else:
            print(usage())
            sys.exit(1)
    except:
        # source_path = ['dataset/2006/', 'dataset/2006-B/']
        source_path = ['test_first_extraction/test/a/', 'test_first_extraction/test/b/']
        # source_path = ['2006-B/']
        folder_level = 2 # suggestion: (n of slashes)-1, taking into account that the path ends with the slash but doesn't start with it

    try:
        start = time.time()

        # print("dir path - ", os.path.dirname(os.path.realpath(__file__)))
        destination_path = get_destination_path(source_path, folder_level)
        print("destination_path: ", destination_path)

        # for path in source_path:
        #     print(get_list_files(path, 'xml'))

        unzip_patents(source_path, destination_path)

        # for path in source_path:
        #     print(get_list_files(path, 'xml'))

        move_patents(source_path, destination_path)

        print("end of unzipping")
        end = time.time()
        print("time: ", end-start)
    except:
        print("ERROR: either calculating destination path or writing index files. Check them out!")

# to do list:
#