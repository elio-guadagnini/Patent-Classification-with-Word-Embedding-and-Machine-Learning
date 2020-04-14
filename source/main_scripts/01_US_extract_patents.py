# -*- coding: utf-8 -*-
import sys
import glob
import os
import stat
import shutil
from io import StringIO, BytesIO, TextIOWrapper
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from lxml import etree

from bs4 import BeautifulSoup

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import txt_data_helper as txth
from helpers import xml_data_helper as xmlh

script_key = "us_extract"
tags = {4 : {
                # BASIC PATENT DATA
                "patent-doc-number" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/doc-number",
                "patent-country" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/country",
                "patent-date" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/date",
                "patent-kind" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/kind",
                "inventors" : "/us-patent-grant/us-bibliographic-data-grant/parties/applicants//applicant/addressbook//*[self::last-name | self::first-name |self::city |self::country] |"
                                + "/us-patent-grant/us-bibliographic-data-grant/us-parties/us-applicants//us-applicant/addressbook//*[self::last-name | self::first-name |self::city |self::country] ",


                # TITLE, ABSTRACT, DESCRIPTIONS AND CLAIMS
                "invention-title": "/us-patent-grant/us-bibliographic-data-grant/invention-title",
                "abstract" : "/us-patent-grant/abstract",
                "description" : "/us-patent-grant/description",
                "claims" : "/us-patent-grant/claims",

                # CLASSIFICATION DATA
                "classification-national-main": "/us-patent-grant/us-bibliographic-data-grant/classification-national/main-classification",
                "classification-ipc":"/us-patent-grant/us-bibliographic-data-grant/classification-ipc/*[self::main-classification | self::further-classification] |" +
                                        "/us-patent-grant/us-bibliographic-data-grant/classifications-ipcr//classification-ipcr/*[self::section | self::class| self::subclass | self::main-group | self::subgroup]",

                "references-cited": "/us-patent-grant/us-bibliographic-data-grant/references-cited//citation/patcit/document-id/doc-number" +
                                        "|/us-patent-grant/us-bibliographic-data-grant/us-references-cited//us-citation/patcit/document-id/doc-number",

                },
2 : {
                "patent-doc-number": "/PATDOC/SDOBI/B100/B110/DNUM/PDAT",
                "patent-country":"/PATDOC/SDOBI/B100/B190/PDAT",
                "patent-date": "/PATDOC/SDOBI/B100/B140/DATE/PDAT",
                "patent-kind": "/PATDOC/SDOBI/B100/B130/PDAT",
                "inventors": "/PATDOC/SDOBI/B700/B720//B721//*[self::FNM | self::SNM | self::CITY | self::CTRY]/PDAT |" +
                             "/PATDOC/SDOBI/B700/B720//B721//*[self::FNM | self::SNM | self::CITY | self::STATE]/PDAT |" +
                             "/PATDOC/SDOBI/B700/B720//B721//*[self::FNM | self::SNM | self::CITY | self::CTRY]/STEXT/PDAT",

                "invention-title": "/PATDOC/SDOAB/BTEXT/PARA/PTEXT/PDAT",
                "abstract": "/PATDOC/SDOAB",
                "description": "/PATDOC/SDODE",
                "claims": "/PATDOC/SDOCL",

                "classification-national-main": "/PATDOC/SDOBI/B500/B520/B521/PDAT",
                "classification-ipc":"/PATDOC/SDOBI/B500/B510/B511/PDAT | " +
                                       "/PATDOC/SDOBI/B500/B510//B512/PDAT",

                "references-cited":"/PATDOC/SDOBI/B500/B560//B561/PCIT/DOC/DNUM/PDAT"
}}

def get_lang(tree):
    try:
        return tree.getroot().attrib["lang"].upper()
    except:
        return None

####__________GET_PATENTS__________####
def get_patents(content, year):
    start_tag = "<?xml"
    start_index = content.find(start_tag, 0)
    patents = []

    # while all the patents haven't been found (indicated by start value)
    while True:
        # find the next starting tag ("<?xml")
        start = content.find(start_tag, start_index)

        # if the next starting tag cannot be found all patents in the xml file have been found.
        # return the list of all the patents contained in the xml file
        if start == -1:
            return list(map(lambda patent : (patent, 4 if year > 2004 else 2), patents))

        # if the next start tag was found, find the next start tag and save the value to end.
        #  append the text from start to end. then set the start to end.
        else:
            end = content.find(start_tag, start_index + len(start_tag) + 1)
            if end == -1:
                end = len(content)
            patents.append(content[start : end])
            start_index = end

# removing also the text of some tags
# def soup_remove_tags(string, tags):
#     soup = BeautifulSoup(string)
#     for tagname in tags:
#         for tag in soup.findAll(tagname):
#             contents = tag.contents
#             parent = tag.parent
#             tag.extract()
#     return soup.get_text()

def re_strip_tags(string): # keeping the text, here i can separate the texts with a whitespace!
    """Returns the given HTML with all tags stripped."""
    return re.sub(r'<[^>]*?>', ' ', string)

####__________ORGANIZE_PROCESSED_PATENT__________####
def organize_processed_patent(patent, dtd_version):
    new_patent = {}
    # if the patent does not have an ipc-classification it cannot be used for
    # classification and is therefore removed and no longer processed
    if ("classification-ipc" not in patent.keys() or
        "claims" not in patent.keys() or
        "description" not in patent.keys()):
        return None
    try:
        # go through all the values for each tag name of the patent
        for tag_name, values in patent.items():
            new_patent[tag_name] = []
            proccesed_values = []
            for val in values:
                # remove newline, empty and None entries
                if (type(val) != str or not re.match("(^\\n)", val)) and val is not None:
                    if re.match("^classification", tag_name) or tag_name == "references-cited":
                        val = re.sub("\s+?", "", val) # remove the whitespaces
                    proccesed_values.append(val)
                    new_patent[tag_name].append(val)
            # save each ipc-classification of the patent as a list of dictionaries. each dictionary containing
            # it's secition, class and subclass value
            if (tag_name == "classification-ipc"):
                if(dtd_version == 2):
                    for value in proccesed_values:
                        if not re.match("^[A-Z].*", value):
                            return None
                values_text=th.get_string_from_list(th.tokenize_text(th.get_string_from_list(new_patent[tag_name], '')),'')
                # values_text = "".join("".join(new_patent[tag_name]).split())
                new_patent[tag_name] = list(map(lambda x : {"section": x[0], "class": x[1:3], "subclass": x[3]}, re.findall("([A-H][0-9]{2}[A-Z][0-9]{2,4})", values_text)))

            # save each inventors of the patent as a dictionary containing: firstname,lastname,city,country
            if (tag_name == "inventors"):
                num_elements = len(new_patent[tag_name])
                if num_elements % 4 != 0:
                    num_elements = num_elements - (num_elements % 4)
                # new_patent[tag_name] = ", ".join(list(map(lambda i : new_patent[tag_name][i] + " " + new_patent[tag_name][i+1], range(0, num_elements, 4))))
                new_patent[tag_name] = th.get_string_from_list(list(map(lambda i : new_patent[tag_name][i] + " " + new_patent[tag_name][i+1], range(0, num_elements, 4))), ', ')

            # save each inventors of the patent as a dictionary containing: firstname,lastname,city,country
            if (tag_name == "references-cited"):
                new_patent[tag_name] = th.get_string_from_list(list(map(lambda element:element, new_patent[tag_name]))), ' ')
                # new_patent[tag_name] = " ".join(list(map(lambda element : element, new_patent[tag_name])))

            # tag names that don't have more than one value are changed from a list to a single value
            if (tag_name in ["invention-title", "classification-national-main", "patent-country", "patent-date", "patent-kind", "patent-doc-number"]):
                try:
                    new_patent[tag_name] = new_patent[tag_name][0]
                except:
                    new_patent[tag_name] = ''

            if (tag_name == "patent-lang"):
                new_patent[tag_name]=th.get_string_from_list(th.tokenize_text(th.get_string_from_list(new_patent[tag_name], '')),'')
                # new_patent[tag_name] = "".join("".join(new_patent[tag_name]).split())
        return new_patent
    except Exception as e:
        print("new error occurred - processsing patent. Error:", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return None

def patent_classifications(patent):
    classifications = ""
    try:
        classifications = list(map(lambda t_classification : t_classification["section"]+t_classification["class"]+t_classification["subclass"], patent["classification-ipc"]))
        if classifications:
            classifications = np.unique(classifications)

            # return classifications, patent["classification-national-main"]
            # alternative
            return th.get_string_from_list(classifications, ' '), patent["classification-national-main"]
        return None, patent["classification-national-main"]
    except:
        return None, patent["classification-national-main"]

####__________PROCESS_PATENT__________####
def process_patent(patent, dtd_version, destination_path, path, patent_index):
    # remove all useless values which might interfere with parsing the xml file
    patent = re.sub("(<\?(?!xml).+?\?>|<b>|<\/b>|<i>|<\/i>|<figref.*?>|<\/figref>|"
        + "<sequence-cwu.*?>|<\/sequence-cwu>|<claim-ref.*?>|<\/claim-ref>)", "", patent)
    patent_data = {}
    try:
        soup = BeautifulSoup(patent, "xml")
        f = StringIO(str(soup.contents[len(soup.contents) - 1]))
        tree = etree.parse(f)

        # print("attributes:", tree.getroot().attrib) # there's lang here!!! for 2 dtd_version

        if dtd_version == 4:
            patent_data["patent-lang"] = get_lang(tree)
        else: # there's no lang in the attributes of dtd_version 2
            patent_data["patent-lang"] = 'EN'

        if patent_data["patent-lang"] != 'EN':
            return None
        else:
            main_classification_flag = True
            # go through all the tag names of a single patent
            for tag_name in tags[dtd_version]:
                # try to get the values for each tag name by using the XPath command specific
                # to the dtd-version of the patent
                values = tree.xpath(tags[dtd_version][tag_name]) # always empty for dtd 2
                # if all the values for the tag name were found, save their text representation as a list
                if not (values == []):
                    if tag_name in ('abstract', 'description', 'claims'):
                        patent_data[tag_name] = [str(etree.tostring(values[0], encoding='utf8', method='xml'), 'utf8')]
                        # print(patent_data[tag_name])
                    elif tag_name == 'classification-national-main' and main_classification_flag:
                        patent_data[tag_name] = list(map(lambda value : value.text.replace(' ', ''), values))
                        main_classification_flag = False
                    else:
                        patent_data[tag_name] = list(map(lambda value : value.text, values))

            # organize the processed patent data
            patent_data = organize_processed_patent(patent_data, dtd_version)
            if (patent_data is not None):
                patent_data["patent-region-doc-number"] = th.get_region_doc_number(patent_data)

                classification_ipc, classification_main = patent_classifications(patent_data)
                filename = th.get_us_filename(patent_data["patent-region-doc-number"])

                patent_data["claims"] = re_strip_tags(patent_data["claims"][0])
                patent_data["description"] = re_strip_tags(patent_data["description"][0])

                tag_list = ['invention-title', 'patent-kind', 'inventors', 'references-cited', 'abstract']
                th.check_xml_variables(patent_data, tag_list)

                patent_data["abstract"] = re_strip_tags(patent_data["abstract"][0])

                xmlh.write_us_xml_patent(destination_path, filename, patent_data["patent-lang"], patent_data["patent-kind"],
                    classification_ipc, patent_data["inventors"],
                    patent_data["invention-title"] + " " + patent_data["abstract"], patent_data["claims"],
                    patent_data["description"], patent_data["references-cited"])

                patent_index.loc[patent_index.shape[0] + 1] = [patent_data["patent-region-doc-number"], patent_data["patent-country"], patent_data["patent-kind"], patent_data["patent-date"], path]

                return classification_ipc
    except Exception as e:
        print("new parsing error occurred - processing data. Error:", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return None

def explore_patents(source_path, destination_path):
    """ explore_patents """
    patent_index = pd.DataFrame(columns=['file', 'country', 'kind', 'date', 'path'])
    for index, path in enumerate(source_path):
        files = fh.get_list_files(path, 'xml')
        # for i in tqdm(range(len(files))):
        for path_filename in files:
            try:
                f = open(path_filename, "rb")
                patent_file = TextIOWrapper(BytesIO(f.read()))

                patents = get_patents(patent_file.read(), int(re.search("([0-9]{4})", path).group()))

                # results = [(process_patent(patents[i][0], patents[i][1], destination_path[index], path, patent_index)) for i in tqdm(range(len(patents)))]
                # alternative
                results = list(map(lambda i : (process_patent(patents[i][0], patents[i][1], destination_path[index], path, patent_index)), tqdm(range(len(patents)))))
            except:
                print("WARNING!!!! Check out the file: ", path_filename)
                continue
    xmlh.write_index(patent_index, script_key)

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

        # source_path = ['test_second_extraction/after 2004/2015/']
        # source_path = ['test_second_extraction/after 2004/2098/'] # test of an error
        # source_path = ['test_second_extraction/after 2004/2097/'] # test of missing abstract and claims, description
        # source_path = ['test_second_extraction/after 2004/2096/'] # test of complex abstract, claims, description
        source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_second_extraction/before 2004/2002/']
        # source_path = ['/Users/elio/Desktop/Patent-Classification/data/test_second_extraction/from 2005/2015/']
        folder_level = 8

    try:
        start = time.time()

        # print("dir path - ", os.path.dirname(os.path.realpath(__file__)))
        destination_path = fh.get_destination_path(source_path, folder_level, script_key)
        # print("destination path: ", destination_path)

        # for path in source_path:
        #     print(fh.get_list_files(path,'xml'))

        explore_patents(source_path, destination_path)

        print("\nend of extraction")
        end = time.time()
        print("time: ", end-start)
    except:
        print("ERROR: during calculating destination path or writing index files. Check them out!")

# to_do_list:
# !!!!!! keep the original dataset
#
# there must be the year of the patent in the path of it - just once!
