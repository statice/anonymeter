# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.

from setuptools import setup, find_packages

requirements = [
    'scikit-learn==1.1.1',
    'numpy==1.22.4',
    'pandas==1.4.3',
    'numexpr==2.8.3',
    'joblib==1.1.0',
    'numba==0.55.2',
    'python-stdnum==1.11',
    'regex==2022.6.2',
    'matplotlib==3.5.2',
    'seaborn==0.11.2',
]

extras = {
    'dev': [
        'flake8==4.0.1', 'flake8-mypy==17.8.0', 'flake8-isort==4.1.1',
        'isort==5.10.1', 'flake8-docstrings==1.6.0', 'jupyterlab==3.4.3',
        'yapf==0.32.0', 'pre-commit==2.20.0'
    ],
    'build': [
        'pytest==7.1.2',
        'pytest-cov==3.0.0',
        'mypy==0.961',
        'pytest-mypy==0.9.1',
    ]
}

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License"
]

setup(
    name='anonymeter',
    version='0.0',
    description=
    'Measure singling out, linkability, and inference risk for synthetic data.',
    author='Statice GmbH',
    author_email='hello@statice.ai',
    url='',
    install_requires=requirements,
    python_requires='<3.10, >3.7',
    classifiers=classifiers,
    packages=find_packages(),
    extras_require=extras,
)
