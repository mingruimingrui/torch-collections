import setuptools

setuptools.setup(
    name='torch-collections',
    version='1.0',
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
    ]
)
