from setuptools import setup, find_packages

setup(
    name='mssakit',  # Required
    version='v1.0.0-paper1',          # Required
    author='Toon Bense',       # Optional
    description='Adjusted mssakit',  # Optional

    url='https://github.com/tbense/MSSAkit',  # Optional
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[         # List your package dependencies here
        'numpy',               # Example dependency
        'requests',            # Example dependency
    ],
)
