import argparse
import os
import subprocess
import sys


def build_docs(version_number: str):
    """Build docs for specific version number

    Args:
        version_number (str): version number without "v" prefix
    """
    print(f"Building docs for version v{version_number}")
    if version_number[0] == 'v':
        raise ValueError(
            'Please provide version number without "v" prefix')
    curr_dir_path: str = os.path.dirname(os.path.realpath(__file__))
    python_path: str = sys.executable
    output = subprocess.check_output(
        f"{python_path} -m sphinx.cmd.build -M html source {curr_dir_path}/serve/v{version_number}"
    )
    print(output.decode())
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
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Script for building package documentation"
    )
    parser.add_argument('version_number')
    args = parser.parse_args()
    build_docs(args.version_number)


if __name__ == "__main__":
    main()
