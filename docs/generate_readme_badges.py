import os
import pathlib
import shutil
import sys

from helpers import run_command

curr_dir_path: str = pathlib.Path(os.path.realpath(__file__)).parent


# Documentation will be automatically generated on new version release
def generate_tests_coverage_badge(python_path: str, cwd: str):
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


def generate_test_badge(python_path: str, cwd: str):
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


def generate_flake8_badge(python_path: str, cwd: str):
    run_command(
        f"{python_path} -m flake8 ./rulekit --exit-zero --format=html --htmldir ./docs/reports/flake8 --statistics --tee --output-file ./docs/reports/flake8/flake8stats.txt",
        cwd=cwd
    )
    run_command(
        "genbadge flake8 -i ./docs/reports/flake8/flake8stats.txt -o ./docs/badges/flake8-badge.svg",
        cwd=cwd
    )


def main():
    print('Updating badges')
    cwd = pathlib.Path(curr_dir_path / '..')  # root repo dir
    python_path: str = sys.executable

    generate_tests_coverage_badge(python_path, cwd)
    generate_test_badge(python_path, cwd)
    generate_flake8_badge(python_path, cwd)

    print('Badges generated successfully!')


if __name__ == "__main__":
    main()
