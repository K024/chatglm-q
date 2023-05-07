from setuptools import setup, find_packages


def my_test_suite():
    import unittest
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


setup(
    name="chatglm-q",
    version="0.0.1-alpha0",
    author="K024",
    description="Another ChatGLM implementation for optimized quantization",
    url="https://github.com/K024/chatglm-q", 

    packages=[
        "chatglm_q",
    ],

    test_suite="setup.my_test_suite",

    install_requires=[
        "tqdm",
        "safetensors",
        "sentencepiece",
        "huggingface_hub",
    ],
)
