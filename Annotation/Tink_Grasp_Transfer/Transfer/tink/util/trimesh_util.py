import argparse
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import trimesh
import json


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
