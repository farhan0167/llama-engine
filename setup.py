from setuptools import setup, find_packages

setup(
    name='llama-engine',
    version='0.6',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package needs here
        "transformers", "accelerate", "bitsandbytes", "torch", "huggingface-hub"
    ],
)
