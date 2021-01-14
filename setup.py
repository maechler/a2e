from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='a2e',
    version='1.0.0',
    author='Markus Mächler',
    description='A library providing AutoML functionality for Keras models focused on autoencoders.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/maechler/a2e",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
