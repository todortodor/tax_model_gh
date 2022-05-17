#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
from tqdm import tqdm
from labellines import labelLines
# import treatment_funcs as t
import treatment_funcs_agri_ind_fe as t

#%% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

#%% Historical plots - Set up
dir_num = 8
carb_cost = 1e-4

sol_all = {}
for y in range(1995,2019):
    print(y)
    sol_all[y] = t.sol(y,dir_num,carb_cost)

print('Computing trade data historically')

traded = {}
tot = {}
# local = {}

for y in sol_all.keys():
    print(y)
    # y = 2018

    cons = sol_all[y].cons
    iot = sol_all[y].iot

    cons_traded = cons.reset_index()
    cons_traded = cons_traded.loc[cons_traded['row_country'] != cons_traded['col_country']]
    cons_traded = cons_traded.set_index(['row_country', 'row_sector', 'col_country'])
    cons_traded = pd.concat({'cons': cons_traded}, names=['col_sector']).reorder_levels([1, 2, 3, 0])

    iot_traded = iot.reset_index()
    iot_traded = iot_traded.loc[iot_traded['row_country'] != iot_traded['col_country']]
    iot_traded = iot_traded.set_index(['row_country', 'row_sector', 'col_country', 'col_sector'])

    # cons_local = cons.reset_index()
    # cons_local = cons_local.loc[cons_local['row_country'] == cons_local['col_country']]# = 0
    # cons_local = cons_local.set_index(['row_country', 'row_sector', 'col_country'])
    # cons_local = pd.concat({'cons':cons_local}, names=['col_sector']).reorder_levels([1,2,3,0])

    # iot_local = iot.reset_index()
    # iot_local = iot_local.loc[iot_local['row_country'] == iot_local['col_country']]
    # iot_local = iot_local.set_index(['row_country', 'row_sector', 'col_country','col_sector'])

    traded[y] = pd.concat([iot_traded, cons_traded])
    # local[y] = pd.concat([iot_local,cons_local])
    tot[y] = pd.concat(
        [sol_all[y].iot, pd.concat({'cons': sol_all[y].cons}, names=['col_sector']).reorder_levels([1, 2, 3, 0])])

#%% Historical traded emissions in baseline and counterfactual
# ATTENTION: co2_prod emissions are in tonnes of CO2 BUT emission intensities are in tonnes / $Mio
# So when multiplying with values (in $Mio) we obtain TONNES
# DIVIDE OBTAINED EMISSIONS BY 1e6 so that everything is in MIO TONNES OF CO2

trade_em = {}

for y in sol_all.keys():
    print(y)
    # y = 2018
    co2_intensity = sol_all[y].co2_intensity.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector'])
    co2_intensity = co2_intensity.T.reset_index().drop(('index', ''), axis=1).T
    # Convert intensities in Mio T / $Mio
    co2_intensity = co2_intensity / 1e6
    co2_intensity.columns = ['co2_int']
    price_hat = sol_all[y].res.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector']).price_hat
    co2_prod = sol_all[y].co2_prod.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector'])
    co2_prod.columns = ['value_co2', 'new_co2']

    trade_em[y] = traded[y].groupby(level=[0,1,2]).sum().join(co2_intensity).join(price_hat).join(co2_prod)
    trade_em[y]['value'] = trade_em[y].value * trade_em[y].co2_int
    trade_em[y]['new'] = trade_em[y].new * trade_em[y].co2_int / trade_em[y].price_hat
    trade_em[y]['value_share'] = trade_em[y].value / trade_em[y].value_co2
    trade_em[y]['new_share'] = trade_em[y].new / trade_em[y].new_co2

#%% Calculate historical series to be plotted
years = [y for y in range(1995,2019)]
change_lv = []
change_pc = []
chshare_pp = []
chshare_pc = []

share_value = []
share_new = []

for y in range(1995,2019):
    print(y)
    change_lv.append(trade_em[y].sum().new - trade_em[y].sum().value)
    change_pc.append((trade_em[y].sum().new - trade_em[y].sum().value)/trade_em[y].sum().value*100)
    chshare_pp.append((trade_em[y].sum().new / trade_em[y].sum().new_co2)*100 - (trade_em[y].sum().value / trade_em[y].sum().value_co2)*100)
    chshare_pc.append(((trade_em[y].sum().new / trade_em[y].sum().new_co2) - (trade_em[y].sum().value / trade_em[y].sum().value_co2)) / (trade_em[y].sum().value / trade_em[y].sum().value_co2) *100)
    share_value.append((trade_em[y].sum().value / trade_em[y].sum().value_co2)*100)
    share_new.append((trade_em[y].sum().new / trade_em[y].sum().new_co2) * 100)

#%% Plots
print('Plotting change in traded emissions in level')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,[x / 1e3 for x in change_lv],label='Reduction in traded emissions',color = sns.color_palette()[5],lw=lw)

ax.set_ylabel('Gigatons of CO2',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)

plt.show()

#%%
print('Plotting change in traded emissions in %')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,change_pc,label='Reduction in traded emissions',color = sns.color_palette()[5],lw=lw)

ax.set_ylabel('% of initial emissions',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)

plt.show()

#%%
print('Plotting change in share of traded emissions in total emissions in p.p.')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,chshare_pp,label='Reduction in traded emissions',color = sns.color_palette()[5],lw=lw)

ax.set_ylabel('Percentage points',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)

plt.show()

#%%
print('Plotting change in share of traded emissions in total emissions in %')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,chshare_pc,label='Reduction in traded emissions',color = sns.color_palette()[5],lw=lw)

ax.set_ylabel('% of initial emission share',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)

plt.show()

#%%
print('Plotting historical evolution of share of traded emissions in total emissions in %')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,share_value,label='Baseline',color = sns.color_palette()[5],lw=lw)
ax.plot(years,share_new,label='Counterfactual',color = sns.color_palette()[2],lw=lw)

ax.set_ylabel('% of total emissions',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
ax.legend()

plt.show()