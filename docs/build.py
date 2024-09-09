import argparse
import os
import shutil
import subprocess
import sys

curr_dir_path: str = os.path.dirname(os.path.realpath(__file__))


def build_docs_using_sphinx(version_number: str):
    print(f"Building docs for version v{version_number}")
    if version_number[0] == 'v':
        raise ValueError(
            'Please provide version number without "v" prefix')

    python_path: str = sys.executable
    output_path: str = f'{curr_dir_path}/serve/v{version_number}'
    output = subprocess.check_output(
        f"{python_path} -m sphinx.cmd.build -M html source {output_path}"
    )
    print(output.decode())

    tmp_path: str = f'{output_path}@'
    shutil.move(output_path, tmp_path)
    shutil.move(os.path.join(tmp_path, 'html'), output_path)
    shutil.rmtree(tmp_path)


def update_index_html(version_number: str):
    with open(os.path.join(curr_dir_path, 'serve', 'index.html'), 'r', encoding='utf-8') as index_html:
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
    with open(os.path.join(curr_dir_path, 'serve', 'index.html'), 'w', encoding='utf-8') as index_html:
        index_html.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Script for building package documentation"
    )
    parser.add_argument('version_number')
    args = parser.parse_args()

    build_docs_using_sphinx(args.version_number)
    update_index_html(args.version_number)


if __name__ == "__main__":
    main()
