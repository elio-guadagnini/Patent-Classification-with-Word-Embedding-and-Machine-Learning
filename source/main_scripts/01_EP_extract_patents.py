# -*- coding: utf-8 -*-
import sys
import glob
import os
import stat
from io import StringIO, BytesIO
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from lxml import etree

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import xml_data_helper as xmlh

script_key = "eu_extract"

# also maths, img, tables and chemistry are considerated due to the text that can follow the tags
def remove_tags(node):
    etree.strip_tags(node, etree.Comment)
    etree.strip_tags(node,'b','br', 'ul', 'li', 'sup', 'sub', 'i', 'u', 'dl', 'dt', 'dd', 'tables', 'table', 'tgroup', 'tbody', 'row', 'entry', 'maths', 'math', 'mrow', 'mi', 'mo', 'mstyle', 'mn', 'msub', 'chemistry', 'img')

def get_nested_text(node):
    text = ""
    if not node.tag is etree.Comment:
        remove_tags(node)
        text = th.get_node_value(node)
        if text == None:
            return ""
    return text

def get_alternative_title(node):
    text = list(map(lambda sub_node : th.get_node_value(sub_node.getnext()), filter(lambda sub_node : th.get_node_value(sub_node).upper() == 'EN', node.iter('B541'))))
    if text:
        if None in text:
            text.remove(None)
        return th.get_string_from_list(text, " ")
    return ""

def get_title(node):
    text = ""
    for sub_node in node.iter('B541'):
        if th.get_node_value(sub_node).upper() == 'EN':
            text = th.get_node_value(sub_node.getnext())
    return text

def get_alternative_abstract(node):
    text = list(map(lambda abst : get_nested_text(abst), node))
    if text:
        if None in text:
            text.remove(None)
        return th.get_string_from_list(text, " ")
    return ""

def get_abstract(node):
    text = ""
    for abst in node:
        # switch all p, heading
        temp = get_nested_text(abst)
        text += temp
    return text

def get_alternative_applicant(patent_document):
    applicant = ""
    sdobi = patent_document[0]
    text = list(map(lambda node : th.handle_ending_node(node, 'snm'), sdobi.iter('B711')))
    if text:
        if None in text:
            text.remove(None)
        text = th.get_flat_list(text)
        return th.get_string_from_list(text, ",")
    return ""

def get_applicant(patent_document):
    applicant = ""
    sdobi = patent_document[0]
    for sub_node in sdobi.iter('B711'):
        # snm, iid (number of opponent), irf (), adr
        for appl in sub_node:
            if appl.tag == 'snm':
                if len(applicant) != 0:
                    applicant += "," + th.get_node_value(appl)
                else:
                    applicant += th.get_node_value(appl)
    return applicant

def get_alternative_citations(patent_document):
    citations = ""
    sdobi = patent_document[0]
    citations = list(map(lambda node : th.handle_citation_node(node, 'B565EP', 'B561'), sdobi.iter('B560')))
    if citations:
        if None in citations:
            citations.remove(None)
        citations = th.get_flat_super_list(citations)
        citations = th.unique_list(citations)
        return th.get_string_from_list(citations, " ")
    return ""

def get_citations(patent_document): # B500, B550, B560
    citations = ""
    sdobi = patent_document[0]
    for node in sdobi.iter('B560'):
        # snm, iid (number of opponent), irf (), adr
        for sub_node in node:
            if sub_node.tag == 'B561':
                for cit in sub_node:
                    if cit.tag == 'text':
                        if len(citations) != 0:
                            temp_citation = th.get_node_value(cit)
                            temp_citation = temp_citation.replace(" ", "")
                            temp_citation = temp_citation[temp_citation.rfind('-')+1:]
                            citations += " " + temp_citation
                        else:
                            temp_citation = th.get_node_value(cit)
                            temp_citation = temp_citation.replace(" ", "")
                            temp_citation = temp_citation[temp_citation.rfind('-')+1:]
                            citations += temp_citation
            elif sub_node.tag == 'B565EP':
                for cit in sub_node:
                    if cit.tag == 'date':
                        if len(citations) != 0:
                            citations += " " + th.get_node_value(cit)
                        else:
                            citations += th.get_node_value(cit)
    return citations

def get_alternative_classcode(patent_document):
    classcode = ""
    sdobi = patent_document[0]

    classcode = list(map(lambda node : th.handle_class_node(node, 0, 4, 'text'), sdobi.iter('classification-ipcr')))

    if classcode:
        if None in classcode:
            classcode.remove(None)
        if "" in classcode:
            classcode.remove("")

        classcode = th.get_flat_list(classcode)
        classcode = th.unique_list(classcode)
        classcode = classcode[:4]
        classcode = th.get_string_from_list(classcode, " ")

    if classcode == "":
        classcode = list(map(lambda sub_node : th.handle_class_node(sub_node, 1, 5, 'text'), sdobi.iter('B510')))

        if classcode:
            if None in classcode:
                classcode.remove(None)
            if "" in classcode:
                classcode.remove("")

            classcode = th.get_flat_list(classcode)
            classcode = th.unique_list(classcode)
            classcode = classcode[:4]
            classcode = th.get_string_from_list(classcode, " ")
    return classcode

def get_classcode(patent_document):
    classcode = ""
    sdobi = patent_document[0]
    for index, sub_node in enumerate(sdobi.iter('classification-ipcr')):
        if len(classcode.split()) == 4:
            break
        for class_ in sub_node:
            if class_.tag == 'text':
                temp = th.get_node_value(class_)
                temp = temp.replace(" ", "")
                temp = temp.replace("/", "")
                if len(temp) > 3:
                    if classcode == "":
                        classcode = temp[:4]
                    elif temp[:4] not in classcode:
                        classcode += " " + temp[:4]
    if classcode == "":
        for sub_node in sdobi.iter('B510'):
            for class_ in sub_node:
                temp = th.get_node_value(class_)
                temp = temp.replace(" ", "")
                temp = temp.replace("/", "")
                if len(temp) > 3:
                    if classcode == "":
                        classcode = temp[1:5]
                    elif temp[1:5] not in classcode:
                        classcode += " " + temp[1:5]
    return classcode

def get_alternative_description(node):
    text = list(map(lambda desc : get_nested_text(desc), node))
    if text:
        if None in text:
            text.remove(None)
        return th.get_string_from_list(text, " ")
    return ""

def get_description(node):
    text = ""
    for desc in node:
        # switch all p, heading
        t = get_nested_text(desc)
        text += " " + t
    return text

def get_claim_type(attributes):
    claim_type = ""
    if 'claim-type' in attributes:
        return attributes['claim-type']
    return claim_type

def get_alternative_claim(node):
    text = list(map(lambda claim : get_nested_text(claim), node.iter('claim-text')))
    if text:
        if None in text:
            text.remove(None)
        return get_claim_type(node.attrib), th.get_string_from_list(text, " ")
    return get_claim_type(node.attrib), ""

def get_claim(node):
    text = ""
    for claim in node.iter('claim-text'):
        text += get_nested_text(claim)
    return get_claim_type(node.attrib), text

def get_file_number(file):
    index = file.find('.')
    if index != -1:
        if index == len(file) - 2 or (len(file) == 12 and index == len(file) - 4):
            return file[:index]
        else:
            return file[2:-8]
    print("WARNING!!! - not found any dot in the file number (attributes)")
    return file

def get_id_document(country, file, kind):
    file = get_file_number(file)
    if kind == 'A1':
        return country + file + "NW" + "B1"
    elif kind == 'B1':
        return country + file + "NW" + "A1"

def get_alternative_country(patent_document):
    country = ""
    sdobi = patent_document[0]

    countries = list(map(lambda node : th.handle_ending_node(node, 'ctry'), sdobi.iter('B330')))

    if countries:
        if None in countries:
            countries.remove(None)

        countries = th.get_flat_list(countries)
        countries = th.unique_list(countries)
        return th.get_string_from_list(countries, " ")
    return ""

def get_country(patent_document):
    country = ""
    sdobi = patent_document[0]
    for sub_node in sdobi.iter('B330'):
        for ctry in sub_node:
            if ctry.tag == 'ctry' and th.get_node_value(ctry) not in country:
                if len(country) != 0:
                    country += " " + th.get_node_value(ctry)
                else:
                    country += th.get_node_value(ctry)
    return country

def a1_parser(patent_document):
    abstract = ""
    for node in patent_document:
        if not node.tag is etree.Comment:
            node_attributes = node.attrib
            if node.tag.upper() == 'SDOBI':
                # add title to abstract
                abstract += get_alternative_title(node)
            elif 'abst' in node_attributes['id'] and node_attributes['lang'].upper() == 'EN':
                # different headings and paragraphs
                abstract += " " + get_alternative_abstract(node)
    applicant = get_alternative_applicant(patent_document)
    citations = get_alternative_citations(patent_document)
    return applicant, abstract, citations

def b1_parser(patent_document, abstract_flag):
    description = ""
    claim_type = ""
    claim = ""
    abstract = ""
    for node in patent_document:
        node_attributes = node.attrib
        if not node.tag is etree.Comment:
            if node.tag.upper() == 'SDOBI':
                abstract += get_alternative_title(node)
            elif 'desc' in node_attributes['id'] and node_attributes['lang'] == 'en':
                # different headings and paragraphs
                description = get_alternative_description(node)
            elif 'claim' in node_attributes['id'] and node_attributes['lang'] == 'en':
                # different claims
                claim_type, claim = get_alternative_claim(node)
            elif abstract_flag == True and 'abst' in node_attributes['id'] and node_attributes['lang'] == 'en':
                # add the title to the abstract)
                abstract += " " + get_alternative_abstract(node)
    citations = get_alternative_citations(patent_document)
    return description, claim, abstract, citations

def explore_patents(source_path, destination_path):
    """ explore_patents """
    patent_data = pd.DataFrame(columns=['file', 'country', 'kind', 'date', 'path'])
    for index, path in enumerate(source_path):
        files = fh.get_list_files(path, 'xml')
        for i in tqdm(range(len(files))):
            try:
                path_filename = files[i]
                parsed_xml = etree.parse(path_filename)
                patent_document = parsed_xml.getroot()

                # country, date, doc_n, dtd_ver, file, id, kind, lang, status
                attributes = patent_document.attrib
                if 'lang' in attributes and 'kind' in attributes and 'file' in attributes and 'date-publ' in attributes and 'country' in attributes:
                    lang = attributes['lang'].upper()
                    kind = attributes['kind'].upper()
                    file = attributes['file']
                    date = attributes['date-publ']
                    region = attributes['country']

                    country = get_alternative_country(patent_document)

                    classcode = get_alternative_classcode(patent_document)

                    if lang == 'EN' and classcode != "":
                        filename = th.get_eu_filename(path_filename) # LLDDDDDDDDLLLD.xml
                        if kind == 'A1': # bibliografy - index
                            applicant, abstract, citations = a1_parser(patent_document)
                            # add A1 to index if the abstract is not empty (and the respective B1 is not in the dataset)
                            if abstract != "":
                                xmlh.write_eu_a1_xml_patent(destination_path[index], filename, lang, kind, classcode, applicant, abstract, citations)

                                filenumber = get_file_number(file)
                                patent_data.loc[patent_data.shape[0] + 1] = [region + filenumber, country, kind, date, path]

                        elif kind == 'B1': # text - data
                            id_respective_document = get_id_document(region, file, kind)
                            # if the A1 has the abstract i do not need to save it here
                            if id_respective_document[:-4] in patent_data['file']:
                                description, claim, abstract, citations = b1_parser(patent_document, False)
                            # if there is not, i have also to save the information of the patent
                            elif id_respective_document[:-4] not in patent_data['file']:
                                description, claim, abstract, citations = b1_parser(patent_document, True)
                            xmlh.write_eu_b1_xml_patent(destination_path[index], filename, lang, kind, id_respective_document, classcode, abstract, claim, description, citations)
                            filenumber = get_file_number(file)
                            patent_data.loc[patent_data.shape[0] + 1] = [region + filenumber, country, kind, date, path]
                else:
                    print("WARNING!!! outlier - no patent document: ", path_filename)
            except:
                print("WARNING!!!! Check out the patent: ", path_filename)
                continue
    xmlh.write_index(patent_data, script_key)

if __name__ == '__main__':
    try:
        if len(sys.argv) == 3:
            # source_path = 'test_clean/*/directories - and inside all the patents'
            source_path = sys.argv[1]
            folder_level = sys.argv[2]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path, folder_level = th.handle_complete_args(source_path, folder_level)
        elif len(sys.argv) == 2:
            # source_path = 'test_clean/*/directories - and inside all the patents'
            source_path = sys.argv[1]

            # here the source_path must be passed as a string of the root directory of all data folders!
            source_path, folder_level = th.handle_partial_args(source_path)
        else:
            print(usage())
            sys.exit(1)
    except:
        # A1 folders before the B1 ones!
        # FOLDER LEVEL - indicates at which level you want to create the new folders:
        # less it is, higher is the level of the folders -> 1 - folder_a/__HERE__/, 2 - folder_a/folder_b/__HERE__/
        # suggestion: (n of slashes)-1, taking into account that the path must end with the slash

        source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/unzipped/A/', '/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/unzipped/B/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B1/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B2/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B3/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B4/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B5/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B6/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B7/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B8/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B9/','/Users/elio/Desktop/Patent-Classification/data/test_first_extraction/test_time/B10/']
        folder_level = source_path[0].count('/')-1

    try:
        start = time.time()

        destination_path = fh.get_destination_path(source_path, folder_level, script_key)

        explore_patents(source_path, destination_path)

        print("\nend of extraction")
        end = time.time()
        print("time: ", end-start)
    except:
        print("ERROR: during calculating destination path or writing index files. Check them out!")

# to_do_list:
# -A1 - abstract (it may be avoided) - B1 has claim and descr (i take the abstracts from this if it is missing in the A1 corrispondent)
#
# !!!!!! keep the original dataset
# !!!!!! classification at A1 level -  early classification system (keep the A1 and B1 separated and then merge them so three sets)
# !!!!!! I have avoided the A1 documents that dont have neither abstract (nor the correspoding B1 patent) !!!!!! I avoid the A1 documents if they do not have the abstract and I save the abstract from B1 if the respective A1 is not in the index. Thus, I read A1s before
# !!!!!! I have to save the abstract and general information of a B1 file if the corresponding A1 file does not have them
#
# added the citations! but i still need to get it.
#
# description, abstract and claim improved: there were some words linked as FOUNDground, but not so many
# classcode improved: now there are always 4 classes and they are the first 4 cited in the patent
# time of execution reduced: by using lambda expressions
#
# citations are ready to be used but i will have problems in the next steps, the citations as i know in us_patents are only in B1s. A1s have only the date. Thus, it's commented and it will not be saved