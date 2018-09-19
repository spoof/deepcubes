from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='intentclf',
    version='0.0.1',
    description='Intent Classifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='The MIT License',
    url='https://github.com/RobotVeraDS/intent-classifier',
    author='Robot Vera',
    author_email='sm.svyat@yandex.ru',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],

    packages=['intentclf', 'intentclf.models'],

    keywords='nlp',
    install_requires=["gensim", "sklearn", "pandas", "pymystem3"],
)
