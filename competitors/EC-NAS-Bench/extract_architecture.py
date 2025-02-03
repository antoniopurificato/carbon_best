#!/usr/bin/env python
# coding: utf-8

# ### EC NAS

# In[1]:


import sys
import os
import json
import random
import pickle
import csv
from pathlib import Path
from math import inf
import pandas as pd
from collections import defaultdict
from typing import List
import numpy as np
from extract_architecture_utils import *

random.seed(0)



# In[3]:


pareto_results = load_experiments_full('/home/hl-neumann/Scrivania/ec-nas/EC-NAS-Bench/ecnas/experiments/semoa/7v_SEMOA_Real', 10)

# In[4]:

max_solution, balanced_solution = obtain_max_balanced_from_pareto(pareto_results)


extract_archi_and_save_to_csv(max_solution, balanced_solution)