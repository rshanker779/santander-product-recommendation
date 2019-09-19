from setuptools import setup, find_packages

setup(
    name="santander-kaggle-flask",
    version="1.0.0",
    author="rshanker779",
    author_email="rshanker779@gmail.com",
    description="Flask API wrapper for Santander Kaggle",
    python_requires=">=3.5",
    install_requires=[
        "flask",
        "pre-commit",
        "tqdm",
        "coverage",
        "base",
        "rshanker779_common",
        "santander_kaggle",
    ],
    packages=find_packages(),
    dependency_links=[
        "git+https://rshanker779@github.com/rshanker779/rshanker779_common.git#egg=rshanker779_common"
    ],
)
