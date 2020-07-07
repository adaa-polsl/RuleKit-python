import setuptools
import os
import io

with io.open(f"{os.path.dirname(os.path.realpath(__file__))}\\..\\README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rulekit-adaa",
    version="1.0.0",
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
    python_requires='>=3.6',
    test_suite="tests",
)
