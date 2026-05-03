from setuptools import setup, find_packages

setup(
    name="armbench",
    version="0.1.0",
    description="ML Inference Profiling Framework",
    author="Rohan",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "armbench=armbench.cli:main",
        ],
    },
)