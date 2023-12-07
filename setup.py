from setuptools import setup, find_packages

setup(
    name='RL_Pizza_Delivery',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'pygame',
        'argparse',
        'tensorboardX'
    ],
    author='Alexy Skoutnev, Oluwatito Ebiwonjumi',
    author_email='alexy.a.skoutnev@vanderbilt.edu',
    description='Simplex based controller train by a PPO Reinforcement Learning policy',
)
