#!/usr/bin/env python3
"""Setup script for dbdiagram2sql package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements(filename):
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f.readlines() 
                if line.strip() and not line.startswith("#") and not line.startswith("-r")
            ]
    return []

setup(
    name="dbdiagram2sql",
    version="1.0.0",
    author="David Nguyen",
    author_email="dvnguyen02@email.com",
    description="Fine-tuned Qwen 2.5 Vision Language Model for converting database schema diagrams to JSON",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dvnguyen02/dbdiagram2sql",
    project_urls={
        "Bug Tracker": "https://github.com/dvnguyen02/dbdiagram2sql/issues",
        "Documentation": "https://github.com/dvnguyen02/dbdiagram2sql/docs",
        "Model Hub": "https://huggingface.co/zodiac2525/Qwen2.5-VL-Diagrams2SQL",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "dbdiagram2sql-extract=dbdiagram2sql.inference:main",
            "dbdiagram2sql-train=dbdiagram2sql.train:main",
            "dbdiagram2sql-benchmark=dbdiagram2sql.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dbdiagram2sql": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    keywords=[
        "database", "schema", "diagram", "vision", "language", "model", 
        "qwen", "transformers", "AI", "machine learning", "deep learning",
        "ER diagram", "database design", "JSON schema"
    ],
    zip_safe=False,
)
