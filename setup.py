from setuptools import setup, find_packages

setup(
    name="experiment-strategies",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "yfinance",
    ],
    python_requires=">=3.7",
)
