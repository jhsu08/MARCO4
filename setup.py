"""Setup script for MARCO4."""

from setuptools import setup, find_packages

setup(
    name="marco4",
    version="4.0.0",
    description="Two-level hierarchical architecture for ARC using Dempster-Shafer theory",
    author="MARCO Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "marco4=marco.main:main",
        ],
    },
)
