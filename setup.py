# ============================================
# Package Configuration File (setup.py)
# For: Packaging and distributing the 'sentisync' module
# Author: Ethan Villalovoz
# ============================================

from setuptools import setup, find_packages

# ============================================
# Setup Configuration
# ============================================

setup(
    name='sentisync',                   # Name of the package (used in PyPI and imports)
    version='0.0.0',                    # Initial version of the package
    author='Ethan Villalovoz',          # Author name
    author_email='ethan.villalovoz@gmail.com',  # Contact email

    packages=find_packages(),           # Automatically find all sub-packages
                                        # Expects an __init__.py in each folder

    install_requires=[],                # List of dependencies to install (add as needed)
                                        # Example: ['pandas', 'scikit-learn', 'lightgbm']
    
    # Optional fields (can be added later for more complete metadata):
    # description='Sentiment analysis pipeline for Reddit/YouTube comments',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/yourusername/sentisync',
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    # python_requires='>=3.7',
)
