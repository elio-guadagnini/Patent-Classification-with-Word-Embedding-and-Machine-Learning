# -*- coding: utf-8 -*-
import sys
from lxml import etree
import os
import pandas as pd

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh

script_key = "xml_data_helper"

settings = {"eu_extract" : {
                        "csv_name" : "data/EU_patents_index.csv",
                },

"us_extract" : {
                        "csv_name" : "data/US_patents_index.csv",
},
"clean" : {
                        "csv_name" : ":::::::::", # because this is not the case and you can't have a path with semicolumns
},
"clean_mix": {
                        "csv_name" : ":::::::::" # because this is not the case and you can't have a path with semicolumns
}}

def get_csv_path(script_key):
    current_path = os.path.dirname(os.path.realpath(__file__))
    index = current_path.rfind('/', 0, -1)
    index = current_path.rfind('/', 0, index-1)
    if script_key == ("clean" or "clean_mix"):
        return fh.link_paths(current_path[:index], settings["eu_extract"]["csv_name"]), fh.link_paths(current_path[:index], settings["us_extract"]["csv_name"])
    return fh.link_paths(current_path[:index], settings[script_key]["csv_name"])

def load_index_csv(csv_path):
    return pd.read_csv(csv_path)

def is_in_csv_alternative(id_document, csv_dataframe, column_name):
    return True if not csv_dataframe[csv_dataframe[column_name] == id_document].empty else False

# def is_in_csv(id_document, csv_path):
#     with open(csv_path, 'r') as patent_index:
#         reader = csv.reader(patent_index)
#         for row in reader:
#             if row[1] == id_document:
#                 return True
#     return False

def write_index(patent_data, script_key):
    current_path = os.path.dirname(os.path.realpath(__file__))

    index = current_path.rfind('/', 0, -1)
    index = current_path.rfind('/', 0, index-1)
    current_path = current_path[:index]

    path_to_csv = current_path + settings[script_key]["csv_name"]
    index = path_to_csv.rfind('/')
    fh.create_folder(path_to_csv[:index], script_key)

    if os.path.isfile(path_to_csv):
        os.remove(path_to_csv)
    patent_data.to_csv(path_to_csv, sep=',', header=True)

#########################################################################################################################
#  handling xml files:

def tree_basic_information(root_node, lang, classcode, applicant, abstract):
    classcode_node = etree.SubElement(root_node, 'class-code').text = classcode
    applicant_node = etree.SubElement(root_node, 'applicant').text = applicant
    abstract_node = etree.SubElement(root_node, 'abstract', lang=lang).text = abstract

def tree_text_information(root_node, lang, claim, description):
    claim_node = etree.SubElement(root_node, 'claim', lang=lang).text = claim
    description_node = etree.SubElement(root_node, 'description', lang=lang).text = description

def tree_endings(root_node, destination_path, filename):
    tree = etree.ElementTree(root_node)
    tree.write(destination_path + filename)

# destination_path[index], filename, file, lang, country, kind, date, classcode, applicant, abstract
def write_eu_a1_xml_patent(destination_path, filename, lang, kind, classcode, applicant, abstract, citations):
    root_node = etree.Element('ep-patent-document', file=filename, lang=lang, kind=kind)

    tree_basic_information(root_node, lang, classcode, applicant, abstract)
    citations_node = etree.SubElement(root_node, 'citations', lang=lang).text = citations

    tree_endings(root_node, destination_path, filename)

# destination_path[index], filename, lang, kind, id_respective_document, classcode, abstract, claim, description
def write_eu_b1_xml_patent(destination_path, filename, lang, kind, id_respective_document, classcode, abstract, claim, description, citations):
    root_node = etree.Element('ep-patent-document', file=filename, lang=lang, kind=kind, id_respective_document=id_respective_document)

    classcode_node = etree.SubElement(root_node, 'class-code').text = classcode
    abstract_node = etree.SubElement(root_node, 'abstract', lang=lang).text = abstract
    tree_text_information(root_node, lang, claim, description)
    # citations_node = etree.SubElement(root_node, 'citations', lang=lang).text = citations

    tree_endings(root_node, destination_path, filename)

# destination_path[index], filename, file, lang, kind, date, classcode, applicant, abstract
def write_us_xml_patent(destination_path, filename, lang, kind, classcode, applicant, abstract, claim, description, citations):
    root_node = etree.Element('us-patent-document', file=filename, lang=lang, kind=kind)

    tree_basic_information(root_node, lang, classcode, applicant, abstract)
    tree_text_information(root_node, lang, claim, description)
    citations_node = etree.SubElement(root_node, 'citations', lang=lang).text = citations

    tree_endings(root_node, destination_path, filename)
