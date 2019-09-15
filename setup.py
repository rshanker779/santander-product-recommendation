from setuptools import setup, find_packages

setup(
    name="santander_kaggle",
    version="1.0.0",
    author="rshanker779",
    author_email="rshanker779@gmail.com",
    description="Solution to Santander Product Recommendation Challenge: https://www.kaggle.com/c/santander-product-recommendation/overview/",
    license="MIT",
    python_requires=">=3.5",
    install_requires=["black", "pandas", "lighgbm", "rshanker779_common"],
    packages=find_packages(),
    entry_points={},
    test_suite="tests",
    dependency_links=[
        "git+https://rshanker779@github.com/rshanker779/rshanker779_common.git#egg=rshanker779_common"
    ],
)
