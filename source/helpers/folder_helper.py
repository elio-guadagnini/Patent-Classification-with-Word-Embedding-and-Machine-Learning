# -*- coding: utf-8 -*-
import os
import sys
import glob
import stat

sys.path.append(os.path.abspath('..'))
from helpers import directory

current_path = directory.get_working_directory()

settings = {
"eu_extract" : {
                        "script_name" : "main_scripts/01_EP_extract_patents.py",
                        "old_" : "/unzipped/",
                        "old_2" : ":::::::::", # because this is not the case and you can't have a path with semicolumns
                        "old_3" : ":::::::::",
                        "new_" : "/parsed/",
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
},
"us_extract" : {
                        "script_name" : "main_scripts/01_US_extract_patents.py",
                        "old_" : "/unzipped/",
                        "old_2" : ":::::::::", # because this is not the case and you can't have a path with semicolumns
                        "old_3" : ":::::::::",
                        "new_" : "/us_parsed/",
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
},
"clean" : {
                        "script_name" : "main_scripts/02_clean_patents.py",
                        "old_" : "/parsed/",
                        "old_2" : "/us_parsed/",
                        "old_3" : ":::::::::",
                        "new_" : "/cleaned/",
                        "new_2" : "/us_cleaned/",
                        "new_3" : "/unknown_cleaned/",
},
"clean_mix": {
                        "script_name" : "main_scripts/02_clean_patents.py",
                        "old_" : "/parsed/",
                        "old_2" : "/us_parsed/",
                        "old_3" : ":::::::::",
                        "new_" : "/cleaned_mix/",
                        "new_2" : "/us_cleaned_mix/",
                        "new_3" : "/unknown_cleaned/",
},
"classify" : {
                        "script_name" : "main_scripts/03_classify_patents.py",
                        "old_" : "/cleaned/",
                        "old_2" : "/cleaned_mix/",
                        "old_3" : "/unknown_cleaned/",
                        "new_" : "/saved_training_set/",
                        "new_2" : "/mix_saved_training_set/",
                        "new_3" : "/unknown_saved_training_set/",
},
"us_classify" : {
                        "script_name" : "main_scripts/03_classify_patents.py",
                        "old_" : "/us_cleaned/",
                        "old_2" : "/us_cleaned_mix/",
                        "old_3" : "/unknown_cleaned/",
                        "new_" : "/us_saved_training_set/",
                        "new_2" : "/us_mix_saved_training_set/",
                        "new_3" : "/unknown_saved_training_set/",
},
"fasttext classify" : {
                        "script_name" : "main_scripts/05_fasttext_classify_patents.py",
                        "old_" : ":::::::::",
                        "old_2" : ":::::::::",
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::",
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
},
"lstm classify" : {
                        "script_name" : "main_scripts/06_lstm_classify_patents.py",
                        "old_" : ":::::::::",
                        "old_2" : ":::::::::",
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::",
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
},
"folder_helper": {
                        "script_name" : "helpers/folder_helper.py",
                        "old_" : ":::::::::", # because we don't need it in this case
                        "old_2" : ":::::::::", # because we don't need it in this case
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::", # because we don't need it in this case
                        "new_3" : ":::::::::",
                        "new_3" : ":::::::::",
},
"xml_data_helper": {
                        "script_name" : "helpers/xml_data_helper.py",
                        "old_" : ":::::::::", # because we don't need it in this case
                        "old_2" : ":::::::::", # because we don't need it in this case
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::", # because we don't need it in this case
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
},
"classification_helper": {
                        "script_name" : "helpers/classification_helper.py",
                        "old_" : ":::::::::", # because we don't need it in this case
                        "old_2" : ":::::::::", # because we don't need it in this case
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::", # because we don't need it in this case
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::"
},
"lstm doc2vec" : {
                        "script_name" : "helpers/lstm_doc2vec_helpers.py",
                        "old_" : ":::::::::",
                        "old_2" : ":::::::::",
                        "old_3" : ":::::::::",
                        "new_" : ":::::::::",
                        "new_2" : ":::::::::",
                        "new_3" : ":::::::::",
}}

#########################################################################################################################
# handling directories:

def create_folder(destination_path):
    if not os.path.exists(destination_path):
        index = current_path.rfind('/', 0, -1)
        # index = current_path.rfind('/', 0, current_path.rfind('/', 0, -1)-1)
        os.chmod(link_paths(current_path[:index], settings["folder_helper"]["script_name"]), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        os.makedirs(destination_path)

# this works for every situation, but you should define the folder_level
def add_folder(string, folder_level, script_key):
    tokens = string.split('/')
    if settings[script_key]["old_"] in string:
        return string.replace(settings[script_key]["old_"], settings[script_key]["new_"], 1)
    elif settings[script_key]["old_2"] in string:
        return string.replace(settings[script_key]["old_2"], settings[script_key]["new_2"], 1)
    elif settings[script_key]["old_3"] in string:
        return string.replace(settings[script_key]["old_3"], settings[script_key]["new_3"], 1)
    elif folder_level > len(tokens):
        return string + settings[script_key]["new_3"][1:]
    return '/'.join(s for s in tokens[0:folder_level]) + settings[script_key]["new_"] + '/'.join(s for s in tokens[folder_level:])

def apply_methods_for_destination_path(path, folder_level, script_key):
    path = add_folder(path, folder_level, script_key)
    create_folder(path)
    return path

def get_destination_path(source_path, folder_level, script_key):
    destination_path = list(map(lambda path : apply_methods_for_destination_path(path, folder_level, script_key), source_path))
    return destination_path

# def get_destination_path(source_path, folder_level, script_key):
#     destination_path = []
#     for path in source_path:
#         path = add_folder(path, folder_level, script_key)
#         destination_path.append(path)
#         create_folder(path)
#     return destination_path

def get_list_files(path, extension):
    if extension:
        return glob.glob(path + '*.' + extension)
    return glob.glob(path)

# this works for the USI server environment!
# def add_folder(string, folder_level, keyword):
#     if '/parsed/' in string:
#         return string.replace('/parsed/', '/' + keyword + '/', 1)
#     return '/'.join(s for s in string.split('/')[0:folder_level+2]) + '/' + keyword + '/' + string.split('/')[-1] + '/'
#     ###### here a problem with short paths


# def get_destination_path(source_path, folder_level):
#     destination_path = []
#     for path in source_path:
#         index = path.find('/')
#         if index != -1:
#             path = add_folder(path, folder_level)
#             destination_path.append(path)
#             create_folder(path)
#         else:
#             destination_path.append('parsed/')
#     return destination_path

# def get_destination_path(source_path, folder_level):
#     destination_path = []
#     mixed_destination_path = []
#     for s_path in source_path:
#         index = s_path.find('/')
#         if index != -1:
#             path = add_folder(s_path, folder_level, 'cleaned')
#             destination_path.append(path)
#             create_folder(path)

#             path = add_folder(s_path, folder_level, 'cleaned_mixed')
#             mixed_destination_path.append(path)
#             create_folder(path)
#         else:
#             destination_path.append('cleaned/')
#             mixed_destination_path.append('cleaned_mixed/')
#     return destination_path, mixed_destination_path

def link_paths(root_path, ending_path):
    return os.path.join(root_path, ending_path)

def join_paths(root_path, ending_path):
    path = link_paths(root_path, ending_path)
    create_folder(path)
    return path

def get_root_location(ending_path):
    index = current_path.rfind('/', 0, current_path.rfind('/', 0, -1)-1)
    return join_paths(current_path[:index], ending_path)

def ensure_exists_path_location(path):
    if os.path.exists(path):
        return True
    return False
