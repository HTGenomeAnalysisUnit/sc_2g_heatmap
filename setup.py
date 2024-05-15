from setuptools import setup, find_packages
import pathlib

# The directory containing this file
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='twogroups_heatmap',
    version='0.1',
    description='Create a heatmap of representing intersection between two groups',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HTGenomeAnalysisUnit/sc_2g_heatmap.git',
    author='Giuditta Clerici',
    author_email='giuditta.clerici@external.ght.org',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where='src'),
)
