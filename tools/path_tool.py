import os
def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_abs_path(positive_path):
    return os.path.join(get_project_root(), positive_path)