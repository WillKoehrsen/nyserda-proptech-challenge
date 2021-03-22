from setuptools import find_packages, setup

setup(
    name="nyserda-proptech-challenge",
    version="0.0.1",
    author="Will Koehrsen",
    author_email="wjk68@case.edu",
    description="Code for competing in the NYSERDA proptech energy forecasting challenge",
    url="https://github.com/WillKoehrsen/nyserda-proptech-challenge",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=["src.*"]),
    python_requires=">=3.8",
)
