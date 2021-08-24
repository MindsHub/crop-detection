from setuptools import setup, find_packages

try:
    # manually check if cv2 is importable, adding it to install_requires would create
    # duplicate installations if it were already installed by compiling from source
    import cv2
    _opencv_installed = True
except (ImportError, RuntimeError):
    _opencv_installed = False

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
        # opencv is already installed (maybe my compiling, so prevent installing a duplicate opencv-python)
    ] if _opencv_installed else [
        'tensorflow>=2.4.0',
        'opencv-python',
    ]
)