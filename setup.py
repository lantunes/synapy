from setuptools import setup

setup(name="synapy",
      version="0.1",
      description="Synapy, a synaptic matrix implementation for Python",
      long_description="Synapy is a Python implementation of a synaptic matrix. "
                       "This project represents an ongoing search for new neurobiologically "
                       "inspired learning algorithms and computational techniques. This work "
                       "is inspired by, and based on, Dr. Arnold Trehub's work. Particularly, "
                       "this project is based on his concept of the synaptic matrix. ",
      license="MIT",
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
      ],
      url='http://github.com/lantunes/synapy',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=["synapy"],
      keywords=["synaptic matrix", "synapy", "neurons", "machine learning"],
      install_requires=["numpy >= 1.12.1"])
