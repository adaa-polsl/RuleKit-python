import argparse
import os
import pathlib
import shutil
import sys

from helpers import run_command

curr_dir_path: str = pathlib.Path(os.path.realpath(__file__)).parent


def build_docs_using_sphinx(version_number: str):
    print(f"Building docs for version v{version_number}")
    output_path: str = f'{curr_dir_path}/serve/v{version_number}'
    run_command([
        sys.executable, '-m', 'sphinx.cmd.build', '-M', 'html', 'source', output_path,
    ],
        cwd=curr_dir_path,
        raise_on_error=False
    )
    tmp_path: str = f'{output_path}@'
    shutil.move(output_path, tmp_path)
    shutil.move(pathlib.Path(tmp_path, 'html'), output_path)
    shutil.rmtree(tmp_path)


def update_index_html(version_number: str):
    print('Updating index.html...')
    with open(pathlib.Path(curr_dir_path / 'serve' / 'index.html'), 'r', encoding='utf-8') as index_html:
        content: str = index_html.read()
        content = content.replace('(latest)', '')
        content = content.replace(
            '<!-- LATEST VERSION PLACEHOLDER -->',
            f'''
                <li class="toctree-l1">
                    <a class="reference internal" href="v{version_number}/index.html">
                        v{version_number} (latest)
                    </a>
                </li>
                <!-- LATEST VERSION PLACEHOLDER -->
            '''
        )
    with open(pathlib.Path(curr_dir_path / 'serve' / 'index.html'), 'w', encoding='utf-8') as index_html:
        index_html.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Script for building package documentation"
    )
    parser.add_argument('version_number')
    args = parser.parse_args()

    version_number: str = args.version_number
    # remove "v" prefix if present
    if version_number.startswith('v'):
        version_number = version_number.replace('v', '')

    build_docs_using_sphinx(version_number)
    update_index_html(version_number)
    print('Documentation built successfully!')


if __name__ == "__main__":
    main()
