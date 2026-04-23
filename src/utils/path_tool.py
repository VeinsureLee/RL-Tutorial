"""
This module provides a set of tools for working with file paths.
"""

import os


def get_root_path():
    """
    返回项目根目录的绝对路径。

    重构后本文件位于 ``src/utils/path_tool.py``，需要向上三层才能拿到项目根：
        src/utils/path_tool.py  ->  src/utils/  ->  src/  ->  <project_root>
    """
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(current_file_path))
    )
    return project_root_path

def get_abs_path(relative_path):
    """
    Get the absolute path to a file or directory.
    """
    project_root_path = get_root_path()
    return os.path.join(project_root_path, relative_path)


if __name__ == "__main__":
    print(get_root_path())
    print(get_abs_path('data'))
