#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:36:59 2022

@author: simonl
"""

import pandas as pd
import os

for d in sorted(next(os.walk('results/'))[1]):
    path = 'results/'+d
    runs_old = pd.read_csv(path+'/runs')
    runs_old.to_csv(path+'/runs_local_path.csv',index=False)
    res_path_drop = '/Users/simonl/Documents/taff/tax_model/'
    runs_old['path'] = runs_old.path.str.replace(res_path_drop,'')
    runs_old.to_csv(path+'/runs',index=False)