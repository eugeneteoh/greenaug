from setuptools import find_packages, setup

setup(
    name="greenaug",
    description="GreenAug: Green Screen Augmentation Enables Scene Generalisation",
    author="Eugene Teoh",
    author_email="eugenetwc1@gmail.com",
    url="greenaug.github.io",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "chromakey[torch] @ git+https://github.com/eugeneteoh/chromakey.git@v0.2.1",
        "av",
        "requests",
    ],
    extras_require={"dev": ["pre-commit", "ruff"]},
)
