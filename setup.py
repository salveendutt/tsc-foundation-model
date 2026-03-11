from setuptools import setup, find_packages

setup(
    name="tsc-foundation-model",
    version="0.1.0",
    description="Time Series Classification using Foundation Models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "aeon>=0.8.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "huggingface-hub>=0.20.0",
    ],
    extras_require={
        "timesfm": ["timesfm"],
    },
)
