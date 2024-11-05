import argparse
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Optional

curr_dir_path: str = pathlib.Path(os.path.realpath(__file__)).parent


def run_command(command: str, cwd: Optional[str] = None):
    output = subprocess.check_output(command, cwd=cwd)
    print(output.decode())


def build_docs_using_sphinx(version_number: str):
    print(f"Building docs for version v{version_number}")
    if version_number[0] == 'v':
        raise ValueError(
            'Please provide version number without "v" prefix')

    python_path: str = sys.executable
    output_path: str = f'{curr_dir_path}/serve/v{version_number}'
    run_command(
        f"{python_path} -m sphinx.cmd.build -M html source {output_path}"
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


def update_badges():
    print('Updating badges')
    cwd = pathlib.Path(curr_dir_path / '..')  # root repo dir

    # update tests coverage badge
    python_path: str = sys.executable
    run_command(
        f"{python_path} -m coverage run -m unittest discover ./tests",
        cwd=cwd
    )
    run_command(
        f"{python_path} -m coverage xml -o ./docs/reports/coverage/coverage.xml",
        cwd=cwd
    )
    run_command(
        f"{python_path} -m coverage html -d ./docs/reports/coverage/",
        cwd=cwd
    )
    run_command(
        "genbadge coverage -i ./docs/reports/coverage/coverage.xml  -o ./docs/badges/coverage-badge.svg",
        cwd=cwd
    )

    # update test badge
    test_report_path = pathlib.Path(curr_dir_path / 'reports' / 'junit')
    shutil.rmtree(test_report_path)
    os.makedirs(test_report_path, exist_ok=True)
    run_command(
        f"{python_path} -m junitxml.main --o ./docs/reports/junit/junit.xml",
        cwd=cwd
    )
    run_command(
        f"{python_path} -m junit2htmlreport ./docs/reports/junit/junit.xml ./docs/reports/junit/report.html",
        cwd=cwd
    )
    run_command(
        "genbadge tests -i ./docs/reports/junit/junit.xml -o ./docs/badges/test-badge.svg",
        cwd=cwd
    )

    # update flake8 badge
    run_command(
        f"{python_path} -m flake8 ./rulekit --exit-zero --format=html --htmldir ./docs/reports/flake8 --statistics --tee --output-file ./docs/reports/flake8/flake8stats.txt",
        cwd=cwd
    )
    run_command(
        "genbadge flake8 -i ./docs/reports/flake8/flake8stats.txt -o ./docs/badges/flake8-badge.svg",
        cwd=cwd
    )


def main():
    parser = argparse.ArgumentParser(
        description="Script for building package documentation"
    )
    parser.add_argument('version_number')
    args = parser.parse_args()

    build_docs_using_sphinx(args.version_number)
    update_index_html(args.version_number)
    update_badges()
    print('Done!')


if __name__ == "__main__":
    main()
