import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = ".."
sys.path.append(project_root)

from sleeprnn.data.nsrr_ss import NsrrSS

if __name__ == "__main__":
    NsrrSS()
