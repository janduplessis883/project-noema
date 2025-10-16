from setuptools import setup, find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name='noema',   # Replace with your package name
    version='0.0.1',
    description="NOEMA is a Feedback Intelligence Dashboard designed to analyze, score, and interpret free-text feedback. It combines natural language processing, machine learning, and large language model (LLM) insights to provide actionable understanding of feedback trends, richness, and sentiment over time.",
    packages=find_packages(),  # It will find all packages in your directory
    install_requires=requirements  # This is the key line to install dependencies
)
