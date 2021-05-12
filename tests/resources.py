import requests
import os
import sys
from typing import List

dir_path = os.path.dirname(os.path.realpath(__file__))

CONFIG_URL = 'https://api.github.com/repos/adaa-polsl/RuleKit/contents/adaa.analytics.rules/test/resources/config'
DATA_URL = 'https://api.github.com/repos/adaa-polsl/RuleKit/contents/adaa.analytics.rules/test/resources/data'
REPORTS_URL = 'https://api.github.com/repos/adaa-polsl/RuleKit/contents/adaa.analytics.rules/test/resources/reports'

RESOURCES_DIRECTORY_PATH = f'{dir_path}/resources'
CONFIG_DIRECTORY_PATH = RESOURCES_DIRECTORY_PATH + '/config'
DATA_DIRECTORY_PATH = RESOURCES_DIRECTORY_PATH + '/data'
REPORTS_DIRECTORY_PATH = RESOURCES_DIRECTORY_PATH + '/reports'


def create_folders():
    # create resources dir 
    if not os.path.exists(RESOURCES_DIRECTORY_PATH):
        os.mkdir(RESOURCES_DIRECTORY_PATH)
    # create CONFIG dir
    if not os.path.exists(CONFIG_DIRECTORY_PATH):
        os.mkdir(CONFIG_DIRECTORY_PATH)
    # create DATA dir
    if not os.path.exists(DATA_DIRECTORY_PATH):
        os.mkdir(DATA_DIRECTORY_PATH)   
    # create REPORTS dir
    if not os.path.exists(REPORTS_DIRECTORY_PATH):
        os.mkdir(REPORTS_DIRECTORY_PATH)
        

def download_files(content: List, directory_path: str):
    for item in content:
        file_name = item['name']
        download_url = item['download_url']
        download_response = requests.get(download_url)
        with open(directory_path +'/'+ file_name, 'wb') as file:
            file.write(download_response.content)


def download_resources():
    #create dirs
    create_folders()

    # download config
    config_content = requests.get(CONFIG_URL).json()
    download_files(config_content, CONFIG_DIRECTORY_PATH)

    # download data
    data_content = requests.get(DATA_URL).json()
    download_files(data_content, DATA_DIRECTORY_PATH)

    # download reports
    reports_content = requests.get(REPORTS_URL).json()
    for item_folder in reports_content:
        folder_name = item_folder['name']
        folder_url = item_folder['url']
        folder_directory = REPORTS_DIRECTORY_PATH + '/' + folder_name
        if not os.path.exists(folder_directory) :
            os.mkdir(folder_directory)
        folder_content = requests.get(folder_url).json()
        download_files(folder_content, folder_directory)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        download_resources()
        print('Resources downloaded successfuly')
    else:
        print('Please add paremeter: python resources.py download')  