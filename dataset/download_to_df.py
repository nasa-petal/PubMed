'''
Code written by David Smith 5/9/2021
dacasm@umich.edu

Downloads all the pubmed data files and creates and pickles a dataframe
'''

from bs4 import BeautifulSoup
import os
import requests
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import wget
import argparse
from typing import List
import glob

def get_args_parser():
    """Creates arguments that this main python file accepts
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser('PubMedDownload', add_help=True)
    parser.add_argument('--download', default=True, type=bool,help='Download files from pubmed')    
    parser.add_argument('--num_files', default=-1, type=int,help='How many pubmed files to download -1 for all files')
    parser.add_argument('--save_directory',default='pubmedfiles',type=str, help='folder where to save pubmed files')
    parser.add_argument('--should_pickle', default=True, type=bool,help='Create a dataframe and save it to a pickle file') 
    return parser

def download_files(num_files:int, save_directory:str):
    """Downloads the files and saves it to a directory 

    Args:
        num_files (int): number of files to download 
        save_directory (str): directory to save to.

    Returns:
        List[str]: paths where download is saved
    """
    os.makedirs(save_directory,exist_ok=True)
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
    }

    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    response = requests.get(url, headers)
    soup = BeautifulSoup(response.content, "html.parser")

    a_tags = soup.find_all("a", href=True)

    download_links = []
    download_paths = []

    count_path = 0
    for html_elem in a_tags:
        if(count_path == num_files):
            break
        if (html_elem.string[-2:] == "gz"):
            download_links.append(url + html_elem.string)
            download_paths.append(
                "./pubmedfiles/" + html_elem.string)
            count_path += 1

    count_link = 0
    for link in download_links:
        if (count_link == num_files):
            break
        wget.download(link, "./pubmedfiles/")
        count_link += 1

    return download_paths

# Handle formatting of more complicated data based on tags

def extract_date(elem, dict_name:str):
    """Extracts the date from an xml element

    Args:
        elem (XML): [description]
        dict_name (str): name of string to extract

    Returns:
        [type]: [description]
    """
    day = "" if elem.find("Day") == None else elem.find("Day").text
    month = "" if elem.find("Month") == None else elem.find("Month").text
    year = "" if elem.find("Year") == None else elem.find("Year").text
    date = {dict_name: "{}/{}/{}".format(day, month, year)}
    return (date, False)


def extract_authors(elem):
    """extract the authors from an xml element

    Args:
        elem (xml): xml element

    Returns:
        tuple: containing a dictionary with authors 
    """
    authors = list(elem)
    authors = {"Authors":
               ", ".join(["{} {}"
                          .format((((author.find("LastName") != None) and author.find("LastName").text) or ""),
                                  (((author.find("Initials") != None) and author.find("Initials").text) or "")) for author in authors])}
    return (authors, False)


def extract_mesh_headings(elem):
    """Extract mesh headings that describe the pubmed articles for example animals, plants, etc. 

    Args:
        elem (xml): xml element

    Returns:
        tuple: containing a list of mesh headings
    """
    mesh_heading_list = {"MeshHeadingList": {}}

    for index, mesh_heading in enumerate(list(elem)):
        if (len(list(mesh_heading)) == 0):
            continue

        current_qualifiers = []
        qualifier_list = mesh_heading.findall("QualifierName")
        current_mesh_heading = "MeshHeading {}".format(index)

        mesh_heading_list["MeshHeadingList"].update(
            {
                current_mesh_heading:
                {"DescriptorName": mesh_heading.find("DescriptorName").text}
            })

        if (len(qualifier_list) != 0 and qualifier_list[0] != None):
            for qualifier in qualifier_list:
                current_qualifiers.append(qualifier.text or "")
            mesh_heading_list["MeshHeadingList"][current_mesh_heading]["QualifierNames"] = \
                ", ".join(current_qualifiers)

    return (mesh_heading_list, False)

# Handle data processing based on tag type


def handle_tag_extraction(elem):
    """[summary]

    Args:
        elem ([type]): [description]

    Returns:
        [type]: [description]
    """
    tag = elem.tag

    if (tag == "abstract"):
        abstract_text = {"AbstractText": elem.find("AbstractText").text}
        return (abstract_text, False)

    elif (tag == "DateCompleted"):
        if (elem.find("Year") == None or elem.find("Year").text == None):
            return
        return extract_date(elem, "DateCompleted")

    elif (tag == "DateRevised"):
        if (elem.find("Year") == None or elem.find("Year").text == None):
            return
        return extract_date(elem, "DateRevised")

    elif (tag == "ArticleIdList"):
        article_ids = [
            {child.attrib["IdType"]: child.text} for child in list(elem)
        ]
        return (article_ids, True)

    elif (tag == "AuthorList"):
        return extract_authors(elem)

    elif (tag == "MeshHeadingList"):
        return extract_mesh_headings(elem)

# Open gzipped files and parse XML files then finally store data into dataframe


def scan_files(download_paths:List[str]):
    """Scans the downloaded files and processes

    Args:
        download_paths (List[str]): [description]

    Returns:
        [type]: [description]
    """
    temp_array = []
    paper_count = 0
    total_papers = len(download_paths)
    for path in download_paths:
        print("{}/{}".format(paper_count+1, total_papers))
        temp_object = {}
        with gzip.open(path, "r") as xml_file:
            context = ET.iterparse(xml_file, events=("start", "end"))

            for index, (event, elem) in enumerate(context):
                # Get the root element.
                if index == 0:
                    root = elem

                if event == "end" and elem.tag == "PubmedArticle":
                    temp_array.append(temp_object.copy())
                    temp_object.clear()
                    root.clear()

                if event == "start" and elem != None and len(list(elem)):
                    extraction_results = handle_tag_extraction(elem)

                    if (extraction_results != None):
                        if (extraction_results[1]):
                            for unpacked_dict in extraction_results[0]:
                                temp_object.update(unpacked_dict)
                        else:
                            temp_object.update(extraction_results[0])
        paper_count += 1

    return pd.DataFrame(temp_array)


# Main program
def run_downloader(args: argparse.ArgumentParser):
    """Main code to download pubmed files, saves to a directory, creates and pickles a dataframe containing authors, abstract, and mesh headings.

    Args:
        args (argparse.ArgumentParser): Argument options to parse
    """
    # if (args.download):
    #     num_files = args.num_files 
    #     download_paths = download_files(args.num_files,args.save_directory)

    if (args.num_files):
        download_paths = list(glob.glob(os.path.join(args.save_directory,"*.gz")))
        print(download_paths)
        if(len(download_paths) != 0):
            print("Parsing Files...")
            data_frame = scan_files(download_paths)

    if (args.should_pickle):
        os.makedirs('pickle',exist_ok=True)
        data_frame.to_pickle("pickle/pickled_data.pkl")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    run_downloader(args)