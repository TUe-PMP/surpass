"""Create a reproducible source/notebook release without runtime caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(), re.MULTILINE).group(1)
major, minor, patch = version.split('.')
destination = root.parent/f'surpass_v{major}.{minor}.{patch}.zip'
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        hidden = any(part.startswith('.') for part in relative.parts)
        if (not path.is_file() or (hidden and relative.as_posix() != '.gitignore')
                or '__pycache__' in relative.parts or path.suffix == '.pyc'
                or any(part.endswith('.egg-info') for part in relative.parts)
                or any(part in {'build', 'dist', 'output'} for part in relative.parts)):
            continue
        # Flat project archive: extract directly into the user's project root.
        # The older nested layout made it easy to retain an obsolete root/src.
        archive.write(path, relative)
with ZipFile(destination) as archive:
    assert archive.testzip() is None
print(destination)
