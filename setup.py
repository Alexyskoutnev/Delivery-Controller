from setuptools import setup, find_packages

setup(
    name='RL_Pizza_Delivery',
    version='0.0.1',
    packages=find_packages(),  # This will automatically discover and include all packages
    install_requires=[
        # List your dependencies here
        'numpy',
        'matplotlib',
        'torch',
        'pygame'
    ],
    author='Alexy Skoutnev, Oluwatito Ebiwonjumi',
    author_email='alexy.a.skoutnev@vanderbilt.edu',
    description='Pizza Delivery planning controller with safety properties',
)
