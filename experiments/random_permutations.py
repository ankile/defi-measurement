import numpy as np

import sys

import os
from pool_state import v3Pool
import numpy as np
import matplotlib.pyplot as plt
import random

import json

import pandas as pd
from prisma import Prisma
from tqdm import tqdm, trange

from sqlalchemy import create_engine

from dotenv import load_dotenv

load_dotenv(override=True)


def add_path():
    current_path = sys.path[0]
    sys.path.append(
        current_path[: current_path.find("defi-measurement")]
        + "liquidity-distribution-history"
    )
