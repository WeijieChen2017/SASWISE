from setuptools import setup, find_packages

setup(
    name="saswise",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    python_requires=">=3.9",
) 