import setuptools
import os
import io

current_path = os.path.dirname(os.path.realpath(__file__))

with io.open(f"{current_path}/README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(f"{current_path}/requirements.txt", mode="r", encoding="utf-8") as f:
    required = f.read().splitlines()

version = None
with open(f"{current_path}/rulekit/VERSION.txt", mode="r", encoding="utf-8") as f:
    version = version

setuptools.setup(
    name="rulekit-adaa",
    version=version,
    author="Cezary Maszczyk",
    author_email="cezary.maszczyk@gmail.com",
    description="Comprehensive suite for rule-based learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaa-polsl/RuleKit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
    ],
    include_package_data = True,
    package_data = {
        'jar': ['*.jar'],
    },
    python_requires='>=3.6',
    install_requires=required,
    test_suite="tests",
)
