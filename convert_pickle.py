# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:51:58 2021

@author: rayen
"""

import pandas as pd

pd.read_pickle('compiled_main.pkl').to_pickle('compiled_protocol_4.pkl', protocol=4)