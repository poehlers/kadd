from setuptools import find_packages, setup

HTTPS_GITHUB_URL = ""  # not yet published

requirements = [
    "numpy", "pandas", "pyyaml", "matplotlib", "pillow",
    "torch", "torchvision", "pydicom", "scikit-learn", "kagglehub",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kadd",
    version="0.1.0",
    author="Pieter Oehlers",
    author_email="poehlers@users.noreply.github.com",
    description="RSNA Knee Abnormality Detection — attention-MIL Swin3D "
                 "classification baseline for the Kaggle competition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=HTTPS_GITHUB_URL,
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["kadd=kadd.__main__:main"]},
)
