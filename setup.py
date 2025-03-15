from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='wasteye',
      version="0.0.12",
      description="Le Wagon Data Science project",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests")
