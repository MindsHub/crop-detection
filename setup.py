from setuptools import setup, find_packages

setup(
    name = 'crop-detection',
    version = '1.0',
    packages = find_packages(),
    package_dir = {"": "."},
    package_data = {
        "": ["model*/*", "model*/*/*"]
    },
    install_requires = [
        'tensorflow>=2.4.0',
        'opencv-python'
    ]
)