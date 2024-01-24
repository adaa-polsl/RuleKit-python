"""Module defining CLI
"""
import sys
import urllib.request
import os
import glob
from ._experiment import _ExperimentRunner
from . import RuleKit, __RULEKIT_RELEASE_VERSION__

dir_path = os.path.dirname(os.path.realpath(__file__))

REPOSITORY_URL: str = 'https://github.com/adaa-polsl/RuleKit'


def _download_rulekit_jar():
    release_version = __RULEKIT_RELEASE_VERSION__
    current_rulekit_jars_files: list[str] = glob.glob(
        f"{dir_path}/jar/*-all.jar")
    download_link = f'{REPOSITORY_URL}/releases/download/v{release_version}/'
    jar_file_name = f'rulekit-{release_version}-all.jar'
    download_link += jar_file_name

    if len(current_rulekit_jars_files) > 0:
        old_files_names = [
            os.path.basename(path)
            for path in current_rulekit_jars_files
        ]
        user_input: str = input(
            f'Old RuleKit jar file/files ({old_files_names}) detected,' +
            'do you want to remove it/them? Type "yes" or "no"\n'
        )
        if user_input == 'yes':
            for old_file_path in current_rulekit_jars_files:
                os.remove(old_file_path)
            print('Old files removed.')
        elif user_input != 'no':
            print('I will treat it as no')
    print(f'Downloading jar file: "{jar_file_name}" from: "{download_link}"')

    def show_progress(block_num, block_size, total_size):
        downloaded = int((block_num * block_size / total_size) * 100)
        print(f'\r{downloaded}%', end='\r')

    urllib.request.urlretrieve(
        download_link, f'{dir_path}/jar/{jar_file_name}', show_progress)
    print('Download finished!\nPackage is ready to use.')


def _main():
    if len(sys.argv) > 1 and sys.argv[1] == 'download_jar':
        _download_rulekit_jar()
    else:
        # use rulekit batch CLI
        rulekit = RuleKit()
        rulekit.init()
        _ExperimentRunner.run(sys.argv[1:])


if __name__ == "__main__":
    _main()
