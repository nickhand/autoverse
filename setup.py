from distutils.core import setup
from distutils.util import convert_path
import os

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

def find_packages(base_path):
    base_path = convert_path(base_path)
    found = []
    for root, dirs, files in os.walk(base_path, followlinks=True):
        dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
        relpath = os.path.relpath(root, base_path)
        parent = relpath.replace(os.sep, '.').lstrip('.')
        if relpath != '.' and parent not in found:
            # foo.bar package but no foo package, skip
            continue
        for dir in dirs:
            if os.path.isfile(os.path.join(root, dir, '__init__.py')):
                package = '.'.join((parent, dir)) if parent else dir
                found.append(package)
    return found

# the base dependencies
with open('requirements.txt', 'r') as fh:
    dependencies = [l.strip() for l in fh]
    
setup(name="autoverse", 
      version=find_version("autoverse/version.py"),
      author="Nick Hand",
      maintainer="Nick Hand",
      maintainer_email="nicholas.adam.hand@gmail.com",
      description="Forward modeling the Universe in Python",
      url="http://github.com/nickhand/autoverse",
      zip_safe=False,
      package_dir = {'autoverse': 'autoverse'},
      packages = find_packages('.'),
      license='GPL3',
      install_requires=dependencies
)

