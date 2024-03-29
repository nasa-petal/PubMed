'''
Code written by David Smith 5/9/2021
dacasm@umich.edu

Downloads all the pubmed data files and creates and pickles a dataframe
'''

from bs4 import BeautifulSoup
import os
from dask.dataframe.io.io import from_pandas
from pandas.core.frame import DataFrame
import requests
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import wget
import argparse
import sys
from typing import List
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, process
import dask.dataframe as dd


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
    parser.add_argument('-d', '--download', default=False, type=str2bool,
                        help='Download files from pubmed')
    parser.add_argument('-n', '--num_files', default=-1, type=int,
                        help='How many pubmed files to download -1 for all files')
    parser.add_argument('-s', '--save_directory', default='pubmedfiles',
                        type=str, help='folder where to save pubmed files')
    parser.add_argument('-p', '--should_pickle', default=True, type=str2bool,
                        help='Create a dataframe and save it to a pickle file')
    parser.add_argument('-c', '--n_cpu', default=16, type=int,
                        help='Number of cpus to use when creating the pickled dataframe')
    return parser


def download_files(num_files: int, save_directory: str):
    """Downloads the files and saves it to a directory 

    Args:
        num_files (int): number of files to download 
        save_directory (str): directory to save to.

    Returns:
        List[str]: paths where download is saved
    """
    os.makedirs(save_directory, exist_ok=True)

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


def extract_date(elem, dict_name: str):
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

    mesh_heading_list.update(
        {"MeshHeadingList": str(mesh_heading_list["MeshHeadingList"])})
    return (mesh_heading_list, False)

# Handle data processing based on tag type


def handle_tag_extraction(elem):
    """[summary]

    Args:
        elem ([type]): [description]

    Returns:
        tuple containing:
            dictionary or array of dictionaries: containing tag category and tag text,
            boolean: stating whether or not we are returning an array of dictionaries

    """
    tag = elem.tag

    if (tag == "Abstract"):
        abstract_text = {"AbstractText": elem.find("AbstractText").text}
        return (abstract_text, False)

    elif (tag == "ArticleTitle"):
        title = {"ArticleTitle": elem.text}
        return (title, False)

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


def extract_tags_from_gz(path: str):
    temp_array = list()
    temp_object = dict()
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

            if event == "start" and elem != None and (len(list(elem)) or elem.text):
                extraction_results = handle_tag_extraction(elem)

                if (extraction_results != None):
                    if (extraction_results[1]):
                        for unpacked_dict in extraction_results[0]:
                            temp_object.update(unpacked_dict)
                    else:
                        temp_object.update(extraction_results[0])
    return pd.DataFrame(temp_array)


def scan_files(download_paths: List[str], n_cpu: int = 16):
    """Scans the downloaded files and processes

    Args:
        download_paths (List[str]): [description]

    Returns:
        [type]: [description]
    """
    processed_data = None
    tqdm.pandas()
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        processed_data = list(tqdm(executor.map(extract_tags_from_gz, download_paths, timeout=None, chunksize=2),
                                   desc=f"Processing {len(download_paths)} examples on {n_cpu} cores",
                                   total=len(download_paths)))
    if(len(processed_data) > 1):
        dataframe = dd.concat(processed_data)
    else:
        dataframe = dd.from_pandas(processed_data[0], 1)
    return dataframe

# Main program


def run_downloader(args: argparse.ArgumentParser):
    """Main code to download pubmed files, saves to a directory, creates and pickles a dataframe containing authors, abstract, and mesh headings.

    Args:
        args (argparse.ArgumentParser): Argument options to parse
    """
    data_frame = pd.DataFrame()

    if (args.download):
        print("Downloading files...")
        num_files = args.num_files
        download_paths = download_files(args.num_files, args.save_directory)

    if (args.num_files):
        download_paths = list(
            glob.glob(os.path.join(args.save_directory, "*.gz")))
        if(len(download_paths) != 0):
            print("Parsing Files...")
            data_frame = scan_files(download_paths, args.n_cpu)
            os.makedirs("parsed-CSV", exist_ok=True)
            data_frame.to_csv("parsed-CSV/pubMed*.csv")

    if (args.should_pickle):
        print("Pickling files...")
        os.makedirs('parquet', exist_ok=True)
        if (len(data_frame.columns) == 0):
            try:
                data_frame = dd.read_csv(
                    "parsed-CSV/pubMed*.csv", dtype={"mid": "string", "pubmed": "float64"})
            except FileNotFoundError:
                print("Did not find directory/file ./parsed-CSV/pubMed.csv")
                sys.exit()
        data_frame.to_parquet("parquet/")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    run_downloader(args)
