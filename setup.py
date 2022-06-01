import setuptools
import os
import io

current_path = os.path.dirname(os.path.realpath(__file__))

with io.open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rulekit",
    version='1.7.0.complex',
    author="Cezary Maszczyk",
    author_email="cezary.maszczyk@gmail.com",
    description="Comprehensive suite for rule-based learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaa-polsl/RuleKit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Java",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    include_package_data = True,
    python_requires='>=3.6',
    install_requires=[
        "pandas ~= 1.2.1",
        "scikit-learn ~= 0.24.1",
        "requests ~= 2.25.1",
        "scipy ~= 1.6.0",
        "joblib ~= 1.0.0",
        "JPype1 ~= 1.3.0",
    ],
    test_suite="tests",
    download_url = 'https://github.com/adaa-polsl/RuleKit-python/archive/refs/tags/v1.6.0.tar.gz',
    project_urls={
        'Bug Tracker': 'https://github.com/adaa-polsl/RuleKit-python/issues',
        'Documentation': 'https://adaa-polsl.github.io/RuleKit-python/',
        'Source Code': 'https://github.com/adaa-polsl/RuleKit-python'
    }
)
