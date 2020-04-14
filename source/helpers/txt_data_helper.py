# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from helpers import folder_helper as fh
from helpers import tool_helper as th
from helpers import classification_helper as ch

#########################################################################################################################
# handling txt files:

def txt_basic_information(file, kind, classcode, applicant, abstract):
    file.write(kind + "\n")
    file.write(classcode + "\n")
    file.write(applicant + "\n")
    if abstract != None:
        file.write(th.get_string_from_list(abstract, ' ') + "\n")
    else:
        file.write("\n")

def txt_text_information(file, claim, description):
    file.write(th.get_string_from_list(claim, ' ') + "\n")
    file.write(th.get_string_from_list(description, ' ') + "\n")

def txt_endings(file):
    file.close()

def write_eu_a1_text_patent(destination_path, filename, kind, classcode, applicant, abstract, citations):
    file = open(destination_path + filename[:-4] + ".txt", "w")

    txt_basic_information(file, kind, classcode, applicant, abstract)
    file.write(citations + "\n")
    txt_endings(file)

def write_eu_b1_text_patent(destination_path, filename, kind, id_respective_document, classcode, abstract, claim, description, citations):
    file = open(destination_path + filename[:-4] + ".txt", "w")

    file.write(kind + "\n")
    file.write(classcode + "\n")
    file.write(id_respective_document + "\n")
    # file.write(abstract + "\n")
    txt_text_information(file, claim, description)
    # file.write(citations)
    txt_endings(file)

def write_eu_mix_text_patent(destination_path, filename, kind, classcode, applicant, abstract, claim, description,citations):
    file = open(destination_path + filename[:-4] + ".txt", "w")

    txt_basic_information(file, kind, classcode, applicant, abstract)
    txt_text_information(file, claim, description)
    file.write(citations + "\n")
    txt_endings(file)

def write_us_text_patent(destination_path, filename, kind, classcode, applicant, abstract, claim, description, citations):
    file = open(destination_path + filename[:-4] + ".txt", "w")

    txt_basic_information(file, kind, classcode, applicant, abstract)
    txt_text_information(file, claim, description)
    file.write(citations + "\n")
    txt_endings(file)

#########################################################################################################################
# loading dataframe for classification:

def get_txt_text(file, length):
    for index in range(length):
        text = file.readline()
    return text

def fill_dataframe(data_frame, classifications_df, classcode, abstract, claim, description):
    if classcode != "":
        # shrink the set to only_top_classes? TRUE/FALSE
        classcode = th.cut_down(classcode, 4, ['H', 'B', 'C'], False) # H, B, C
        data_frame.loc[data_frame.shape[0] + 1] = [abstract, claim, description, classcode]
        classifications_df = th.calculate_class_distribution(classcode, classifications_df)

def handle_patent_file(data_frame, classifications_df, path_filename):
    file = open(path_filename, "r")

    kind = get_txt_text(file, 1).strip()
    if kind == 'A1':
        classcode = get_txt_text(file, 1).strip()
        applicant = get_txt_text(file, 1).strip()

        abstract = get_txt_text(file, 1).strip()
        citations = get_txt_text(file, 1).strip()
        file.close()

        fill_dataframe(data_frame, classifications_df, classcode, abstract, None, None)
    elif kind == 'B1':
        classcode = get_txt_text(file, 1).strip()
        id_respective_document = get_txt_text(file, 1).strip()

        # abstract = get_txt_text(file, 1).strip()
        claim = get_txt_text(file, 1).strip()
        description = get_txt_text(file, 1).strip()
        # citations = get_txt_text(file, 1).strip()
        file.close()

        fill_dataframe(data_frame, classifications_df, classcode, None, claim, description)
    else:
        if kind == 'A1B1':
            print("eu_mix_patent !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("us_patent     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        classcode = get_txt_text(file, 1).strip()
        applicant = get_txt_text(file, 1).strip()

        abstract = get_txt_text(file, 1).strip()
        claim = get_txt_text(file, 1).strip()
        description = get_txt_text(file, 1).strip()
        citations = get_txt_text(file, 1).strip()
        file.close()

        fill_dataframe(data_frame, classifications_df, classcode, abstract, claim, description)
    return th.get_patent_id(path_filename)

def handle_path_patent(data_frame, classifications_df, path):
    return list(map(lambda path_filename : handle_patent_file(data_frame, classifications_df, path_filename), fh.get_list_files(path, 'txt')))

def load_data(source_path):
    print('###  reading patents  ###')
    """ load_data """
    data_frame = pd.DataFrame(columns=['abstract', 'claim', 'description', 'classification'])
    classifications_df = pd.DataFrame(columns=['class', 'count'])

    patent_ids = []
    patent_ids = list(map(lambda path : handle_path_patent(data_frame, classifications_df, path), tqdm(source_path)))
    patent_ids = th.get_flat_list(patent_ids)

    data_frame['id'] = data_frame.index
    classifications_df.sort_values(by=['count'], ascending=False, inplace=True, kind='quicksort')
    return patent_ids, data_frame, classifications_df

def get_final_df(patent_ids, temp_df, classif_type):
    classification_target = 'description_claim_abstract_title'

    data_frame = pd.DataFrame(columns=['patent_id', 'text', 'classification'])

    data_frame["text"] = temp_df["abstract"].str.cat(temp_df["claim"], sep =" ", na_rep=" ")
    # data_frame["text"] = temp_df["claim"]
    # data_frame["text"] = temp_df["description"]
    # data_frame["text"] = temp_df["abstract"].str.cat(temp_df["claim"], sep =" ", na_rep=" ").str.cat(temp_df["description"], sep =" ", na_rep=" ")

    classification_types = ch.get_classifications(temp_df)
    data_frame["classification"] = classification_types[classif_type]

    data_frame["patent_id"] = patent_ids
    return data_frame, classification_target, classif_type




#########################################################################################################################
# useless utils:

def load_data_2(source_path):
    print('###  reading patents  ###')
    """ load_data_2 """
    data_frame = pd.DataFrame(columns=['abstract', 'claim', 'description', 'classification'])
    classifications_df = pd.DataFrame(columns=['class', 'count'])
    patent_ids = []
    for path in source_path:
        for patent_index, path_filename in enumerate(fh.get_list_files(path, 'txt')):
            file = open(path_filename, "r")

            patent_ids.append(th.get_patent_id(path_filename))
            kind = get_txt_text(file, 1).strip()

            if kind == 'A1':
                classcode = get_txt_text(file, 1).strip()
                applicant = get_txt_text(file, 1).strip()

                abstract = get_txt_text(file, 1).strip()
                citations = get_txt_text(file, 1).strip()
                file.close()

                fill_dataframe(data_frame, classifications_df, classcode, abstract, None, None, patent_index)
            elif kind == 'B1':
                classcode = get_txt_text(file, 1).strip()
                id_respective_document = get_txt_text(file, 1).strip()

                # abstract = get_txt_text(file, 1).strip()
                claim = get_txt_text(file, 1).strip()
                description = get_txt_text(file, 1).strip()
                # citations = get_txt_text(file, 1).strip()
                file.close()

                fill_dataframe(data_frame, classifications_df, classcode, None, claim, description, patent_index)
            else:
                if kind == 'A1B1':
                    print("eu_mix_patent !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    print("us_patent     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                classcode = get_txt_text(file, 1).strip()
                applicant = get_txt_text(file, 1).strip()

                abstract = get_txt_text(file, 1).strip()
                claim = get_txt_text(file, 1).strip()
                description = get_txt_text(file, 1).strip()
                citations = get_txt_text(file, 1).strip()
                file.close()

                fill_dataframe(data_frame, classifications_df, classcode, abstract, claim, description,patent_index)

    data_frame['id'] = data_frame.index
    classifications_df.sort_values(by=['count'], ascending=False, inplace=True, kind='quicksort')
    return patent_ids, data_frame, classifications_df

def apply_method_for_reading_patents_1_2(data_frame, classifications_df, path):
    return [apply_method_for_reading_patents_2(data_frame, classifications_df, patent_index, path_filename) for patent_index, path_filename in enumerate(fh.get_list_files(path, 'txt'))]

def load_data_3(source_path):
    print('###  reading patents  ###')
    """ load_data_2 """
    data_frame = pd.DataFrame(columns=['abstract', 'claim', 'description', 'classification'])
    classifications_df = pd.DataFrame(columns=['class', 'count'])
    patent_ids = []

    patent_ids = [apply_method_for_reading_patents_1_2(data_frame, classifications_df, path) for path in source_path]
    patent_ids = th.get_flat_list(patent_ids)

    data_frame['id'] = data_frame.index
    classifications_df.sort_values(by=['count'], ascending=False, inplace=True, kind='quicksort')
    return patent_ids, data_frame, classifications_df

def load_path_data(source_path, destination_path):
    print("seeking for patents ...")
    patents = []
    for index, path in enumerate(source_path):
        patent_data = pd.DataFrame(columns=['file_path'])
        for file_path in fh.get_list_files(path, 'xml'):
            patent_data.loc[patent_data.shape[0] + 1] = [file_path]
        patents.append((patent_data, destination_path[index], path))
    return patents

def explore_patents_2(patent_path_dataframe):
    """ explore_patents """
    patent_data = pd.DataFrame(columns=['file', 'country', 'kind', 'date', 'path'])
    for patent_paths, destination_path, source_path in patent_path_dataframe:
        for index, row in tqdm(patent_paths.iterrows(), total=patent_paths.shape[0]):
            try:
                # print("file: ", row['file_path'])
                parsed_xml = etree.parse(row['file_path'])
                patent_document = parsed_xml.getroot()

                # country, date, doc_n, dtd_ver, file, id, kind, lang, status
                attributes = patent_document.attrib
                if 'lang' in attributes and 'kind' in attributes and 'file' in attributes and 'date-publ' in attributes and 'country' in attributes:
                    lang = attributes['lang'].upper()
                    kind = attributes['kind'].upper()
                    file = attributes['file']
                    date = attributes['date-publ']
                    region = attributes['country']

                    # country = get_country(patent_document)
                    country = get_alternative_country(patent_document)

                    # classcode = get_classcode(patent_document)  # LDDLDDD
                    classcode = get_alternative_classcode(patent_document)

                    if lang == 'EN' and classcode != "":
                        filename = th.get_eu_filename(row['file_path']) # LLDDDDDDDDLLLD.xml
                        if kind == 'A1': # bibliografy - index
                            applicant, abstract, citations = a1_parser(patent_document)
                            # add A1 to index if the abstract is not empty (and the respective B1 is not in the dataset)
                            if abstract != "": # and id_respective_document[:-4] not in patent_data['file']:
                                xmlh.write_eu_a1_xml_patent(destination_path, filename, lang, kind, classcode, applicant, abstract, citations)

                                filenumber = get_file_number(file)
                                patent_data.loc[patent_data.shape[0] + 1] = [region + filenumber, country, kind, date, source_path]
                            # # if the respective B1 is in the dataset, save the A1 but dont write it to the index
                            # elif id_respective_document[:-4] in patent_data['file']:
                            #     write_a1_patent(destination_path, filename, file, lang, country, kind, date, classcode, applicant, abstract)

                        elif kind == 'B1': # text - data
                            id_respective_document = get_id_document(region, file, kind)
                            # if the A1 has the abstract i do not need to save it here
                            if id_respective_document[:-4] in patent_data['file']:
                                description, claim, abstract, citations = b1_parser(patent_document, False)
                            # if there is not, i have also to save the information of the patent
                            elif id_respective_document[:-4] not in patent_data['file']:
                                description, claim, abstract, citations = b1_parser(patent_document, True)
                            xmlh.write_eu_b1_xml_patent(destination_path, filename, lang, kind, id_respective_document, classcode, abstract, claim, description, citations)
                            filenumber = get_file_number(file)
                            patent_data.loc[patent_data.shape[0] + 1] = [region + filenumber, country, kind, date, source_path]
                else:
                    print("WARNING!!! outlier - no patent document: ", row['file_path'])
            except:
                print("WARNING!!!! Check out the patent: ", row['file_path'])
                continue
    xmlh.write_index(patent_data, script_key)

# patent_path_dataframe = load_path_data(source_path, destination_path)
# explore_patents_2(patent_path_dataframe)

# always : 1440.18
#    new : 1692.09
#        :  251.91
#        :  +17.50%
