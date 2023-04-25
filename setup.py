from setuptools import setup, find_packages


def my_test_suite():
    import unittest
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


setup(
    name="chatglm-onnx",
    version="0.0.1-alpha0",
    author="K024",
    description="ChatGLM optimized for ONNX export and ONNXRuntime execution",
    url="https://github.com/K024/chatglm-onnx", 

    packages=[
        "chatglm_q",
        "chatglm_onnx",
    ],

    test_suite="setup.my_test_suite",

    install_requires=[
        "tqdm",
        "safetensors",
        "sentencepiece",
    ],
)
