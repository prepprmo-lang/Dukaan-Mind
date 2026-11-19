import os
import runpy
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)

runpy.run_path(os.path.join(SRC_DIR, "app.py"), run_name="__main__")
