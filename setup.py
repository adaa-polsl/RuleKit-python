import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rulekit-adaa", # Replace with your own username
    version="0.0.1",
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
)