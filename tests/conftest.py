import sys
import os


project_root = os.path.dirname(os.path.dirname(__file__))
parent_dir = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
