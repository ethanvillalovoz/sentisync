from setuptools import setup, find_packages

setup(
    name='sentisync',
    version='1.0.0',
    description='Real-time YouTube comment sentiment analysis with Flask, a Chrome extension, and an MLOps training pipeline.',
    author='Ethan Villalovoz',
    author_email='ethan.villalovoz@gmail.com',
    url='https://github.com/ethanvillalovoz/sentisync',
    packages=find_packages(),
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
