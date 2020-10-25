import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_research",
    version="0.0.1",
    author="Mark Tuddenham",
    author_email="mark@tudders.com",
    description="A few helper functions for ML research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarkTuddenham/pytorch_research",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
