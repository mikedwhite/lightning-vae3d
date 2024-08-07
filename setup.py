import setuptools

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='lightning-vae3d',
    version='0.0.3',
    author='Mike White',
    author_email='mike.white@ukaea.uk',
    description='ResNet-like variational autoencoder for encoding 3D volume elements.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/mikedwhite/lightning-vae3d',
    license='Modified BSD',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)