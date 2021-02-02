import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rank-aggregation-rayen96", # Replace with your own username
    version="0.0.1",
    author="Rayen Tan",
    author_email="rayentan96@gmail.com",
    description="A package with rank aggregation algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rayen96/rank_aggregation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)