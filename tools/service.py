import numpy as np
import os


def list_files_and_dirs(startpath, exclude_hidden=True, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = set()
    else:
        exclude_dirs = set(exclude_dirs)

    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        foldername = os.path.basename(root)

        if foldername not in exclude_dirs:
            print(f"{indent}{foldername}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if not (exclude_hidden and f.startswith('.')):
                    print(f"{subindent}{f}")

        if exclude_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]

        dirs[:] = [d for d in dirs if d not in exclude_dirs]


if __name__ == "__main__":
    list_files_and_dirs('..', exclude_dirs=['temp', 'logs', '__pycache__'])
    