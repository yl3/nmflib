from setuptools import setup, find_packages

setup(name='nmflib',
      version='0.0.1',
      description='Mutation NMF analysis library',
      author='Yilong Li',
      author_email='yilong.li.yl3@gmail.com',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy',
          'pandas',
          'progressbar2',
          'pysam',
          'tqdm',
          'joblib',
      ],
      zip_safe=False)
