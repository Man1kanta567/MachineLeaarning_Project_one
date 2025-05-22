from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT ='-e .'

def getModules(requirementTxtPath:str)->List[str]:
    requirements = []
    with open(requirementTxtPath) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements



setup(
    name="Machine_learning_project_one",
    description="End to End machine learning project ",
    author="Manikanta Mukkapati",
    author_email="manikantamukkapti9@gmail.com",
    packages=find_packages(),
    install_requires=getModules('requirements.txt')
)

