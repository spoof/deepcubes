from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deepcubes',
    version='0.2',
    description='Vera deep learning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='The MIT License',
    url='https://github.com/RobotVeraDS/deepcubes',
    author='Robot Vera',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),

    keywords='nlp',
    install_requires=["gensim", "sklearn", "pandas", "pymystem3",
                      "numpy", "flask", "editdistance", "torch"],
)
