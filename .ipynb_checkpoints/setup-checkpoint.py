from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name = "paddletransfer",
    version = "0.0.1",
    author = "Baidu-BDL",
    author_email = "autodl@baidu.com",
    description = "transfer learning toolkits for finetune deep learning models",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9'
    ],
    packages = find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'paddle>=2.2',
        'numpy>=1.20'
    ],
    license = 'Apache 2.0',
    keywords = "transfer learning toolkits for paddle models"
)