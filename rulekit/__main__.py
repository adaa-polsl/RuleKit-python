import sys
import json
from typing import List
import urllib.request
import os
import glob
from rulekit.experiment import ExperimentRunner
from rulekit import RuleKit

dir_path = os.path.dirname(os.path.realpath(__file__))


def download_rulekit_jar():
    release_version = 'latest'
    current_rulekit_jars_files: List[str] = glob.glob(
        f"{dir_path}/jar/*-all.jar")
    url = f"https://api.github.com/repos/adaa-polsl/RuleKit/releases/{release_version}"
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    response = urllib.request.urlopen(req)
    response = json.loads(response.read())
    latest_release_version = response['tag_name']
    print('Fetching latest RuleKit release version: ', latest_release_version)
    assets = response['assets']
    asset = list(
        filter(lambda asset: asset['name'].endswith('-all.jar'), assets))
    if len(asset) != 1:
        raise Exception('Failed to fetch latest RuleKit release jar file.')
    asset = asset[0]
    download_link = asset['browser_download_url']

    if len(current_rulekit_jars_files) > 0:
        old_files_names = list(
            map(lambda path: os.path.basename(path), current_rulekit_jars_files))
        tmp = input(
            f'Old RuleKit jar file/files ({old_files_names}) detected, do you want to remove it/them? Type "yes" or "no"\n')
        if tmp == 'yes':
            for old_file_path in current_rulekit_jars_files:
                os.remove(old_file_path)
            print('Old files removed.')
        elif tmp != 'no':
            print('I will treat it as no')
    print(f'Downloading jar file: "{asset["name"]}" from: "{download_link}"')

    def show_progress(block_num, block_size, total_size):
        downloaded = int((block_num * block_size / total_size) * 100)
        print(f'\r{downloaded}%', end='\r')

    urllib.request.urlretrieve(
        download_link, f'{dir_path}/jar/{asset["name"]}', show_progress)
    print('Download finished!\nPackage is ready to use.')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'download_jar':
        download_rulekit_jar()
    else:
        # use rulekit batch CLI
        rulekit = RuleKit()
        rulekit.init()
        ExperimentRunner.run(sys.argv[1:])


if __name__ == "__main__":
    main()
