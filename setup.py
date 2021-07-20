from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='concentration_lib',
    packages=['concentration_lib'],
    version='0.0.1',
    author="Patrick Saux",
    author_email="patrick.saux@ginria.fr",
    description="Library for basic concentration bounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sauxpa/concentration_lib",
    install_requires=['numpy', 'scipy'],
    python_requires='>=3.6',
)
