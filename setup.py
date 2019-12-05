from setuptools import setup, find_packages

setup(name='mutlib',
      version='0.0.0',
      description='Point mutation analysis library',
      author='Yilong Li',
      author_email='yilong.li.yl3@gmail.com',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      zip_safe=False)
