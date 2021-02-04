import setuptools
import os
import io

with io.open(f"{os.path.dirname(os.path.realpath(__file__))}\\README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(f"{os.path.dirname(os.path.realpath(__file__))}/requirements.txt", mode="r", encoding="utf-8") as f:
    required = f.read().splitlines()
print(required)

setuptools.setup(
    name="rulekit-adaa",
    version="1.2.5",
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
        # And include any *.msg files found in the 'hello' package, too:
        'jar': ['*.jar'],
    },
    python_requires='>=3.6',
    install_requires=required,
    test_suite="tests",
)
