from setuptools import find_packages, setup

setup(
    name="stocks_forecasting_inference",
    version="1.0.0",
    author="Onur Kerimoglu",
    author_email="kerimoglu.o@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
