import setuptools
from Cython.Build import cythonize
import numpy as np

setuptools.setup(
    name='torch-collections',
    version='0.4b',
    description='A collection of deep learning models, and utility tookit written in torch',
    url='https://github.com/mingruimingrui/torch-collections',
    author='Wang Ming Rui',
    author_email='mingruimingrui@hotmail.com',
    packages=[
        'torch_collections',
        'torch_collections.losses',
        'torch_collections.models',
        'torch_collections.modules',
        'torch_collections.utils'
    ],
    include_dirs=[np.get_include()],
    ext_modules=cythonize('torch_collections/utils/cpu_nms.pyx')
)
