from setuptools import setup, find_packages

setup(
    name="ChemEXIN",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", "tensorflow", "simple_colors"
    ],
    # Additional metadata about your package
    author="Dinesh Sharma, Danish Aslam, Kopal Sharma, Aditya Mittal, and B Jayaram",
    author_email="run478@gmail.com",
    description="ChemEXIN:our tool",
    license="SCFBio,IIT Delhi",
    keywords="Exon-intron boundary",
)

