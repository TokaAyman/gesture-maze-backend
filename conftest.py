# File: gesture-maze-backend/conftest.py

import sys
import os

# Insert the project root (the folder containing app/ and tests/) into sys.path.
# This makes `import app.main` work.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
