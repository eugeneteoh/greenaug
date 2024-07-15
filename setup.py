from setuptools import find_packages, setup

setup(
    name="greenaug",
    description="GreenAug: Green Screen Augmentation Enables Scene Generalisation",
    author="Eugene Teoh",
    author_email="eugenetwc1@gmail.com",
    url="https://greenaug.github.io",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "chromakey[torch] @ git+https://github.com/eugeneteoh/chromakey.git@v0.2.1",
        "av",
        "einops",
        "huggingface_hub",
    ],
    extras_require={
        "generative": [
            "diffusers",
            "transformers",
            "accelerate",
            "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git",
        ],
        "mask": [
            "lightning",
            "albumentations",
            "segmentation-models-pytorch",
            "wandb",
            "hydra-core",
        ],
        "dev": ["pre-commit", "ruff"],
    },
)
