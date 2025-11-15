#!/usr/bin/env python3
"""
Setup script for HERBench.

This allows installing HERBench as a package, making imports work correctly.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="herbench",
    version="0.1.0",
    description="HERBench: Benchmarking Multi-Evidence Integration in Video Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HERBench Team",
    python_requires=">=3.8,<3.13",  # Critical: Hydra doesn't support Python 3.13+
    packages=find_packages(where="evaluation"),
    package_dir={"": "evaluation"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "herbench-eval=run_evaluation:main",
            "herbench-accuracy=calculate_accuracy:main",
            "herbench-mrfs=calculate_mrfs:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
