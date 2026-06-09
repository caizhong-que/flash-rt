from setuptools import find_packages, setup

setup(
    name="flash-rt",
    version="1.0.0",
    description="FLASH-RT: Real-Time APT Detection in Provenance Graphs",
    packages=find_packages(exclude=["experiments*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "gensim>=4.3.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "orjson>=3.8.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "psutil>=5.9.0",
        "gdown>=4.6.0",
    ],
)
