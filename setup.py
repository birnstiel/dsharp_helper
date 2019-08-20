"""
Setup file for package `dsharp_helper`.
"""
import setuptools  # noqa
# from numpy.distutils.core import Extension
import pathlib
import warnings

PACKAGENAME = 'dsharp_helper'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    from numpy.distutils.core import setup

    extensions = []

    def setup_function(extensions):
        setup(
            name=PACKAGENAME,
            description='python routines to help download ALMA DSHARP data',
            version='0.0.1',
            long_description=(HERE / "README.md").read_text(),
            long_description_content_type='text/markdown',
            url='https://github.com/birnstiel/dsharp_helper',
            author='Til Birnstiel & DSHARP collaboration',
            author_email='til.birnstiel@lmu.de',
            license='GPLv3',
            packages=[PACKAGENAME],
            package_dir={PACKAGENAME: 'dsharp_helper'},
            package_data={PACKAGENAME: [
                'data/*.*',
                'notebooks/*.ipynb',
                ]},
            include_package_data=True,
            install_requires=[
                'scipy',
                'numpy',
                'matplotlib',
                'astropy',
                'pandas',
                'sphinx',
                'nbsphinx'],
            zip_safe=False,
            ext_modules=extensions
            )
    try:
        setup_function(extensions)
    except BaseException:
        warnings.warn('Could not compile extensions, code will be slow')
        setup_function([])
