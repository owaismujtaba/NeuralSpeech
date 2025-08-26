import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'data', 'raw')
DATA_DIR_PROCED = Path(CUR_DIR, 'data', 'processed')