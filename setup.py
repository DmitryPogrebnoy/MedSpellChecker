import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
print(here / 'PACKAGE.md')
# Get the long description from the README file
long_description = (here / 'PACKAGE.md').read_text(encoding='utf-8')

setup(
    name='medspellchecker',
    version='0.0.4',
    description='Fast and effective spellchecker for Russian medical texts',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DmitryPogrebnoy/MedSpellChecker',
    author='Dmitry Pogrebnoy',
    author_email='pogrebnoy.inc@gmail.com',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing',
        'Topic :: Text Editors :: Word Processors',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
    keywords='spellchecker, nlp, medical, text correction',
    package_dir={'medspellchecker.tool': 'medspellchecker/tool'},
    packages=['medspellchecker.tool'],
    python_requires='>=3.9',
    install_requires=['pymorphy2>=0.9.1',
                      'torch>=1.11.0',
                      'numpy>=1.22',
                      'scipy>=1.7.3',
                      'tqdm>=4.62.3',
                      'nltk>=3.6.7',
                      'editdistpy>=0.1.3',
                      'accelerate>=0.12.0',
                      'transformers>=4.22.2',
                      'sacremoses>=0.0.49',
                      'pynvml>=11.4.1'],
    include_package_data=True,
    package_data={'medspellchecker.tool': ['data/processed_lemmatized_all_dict.txt']},
    project_urls={
        'Bug Reports': 'https://github.com/DmitryPogrebnoy/MedSpellChecker/issues',
        'Funding': 'https://donate.pypi.org',
    },
)
