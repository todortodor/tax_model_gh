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

fixed_carb_tax = True # if True, will load historical data for a given carbon cost
carb_cost = 1e-4
adjust = False  # if True, will adjust for dollar according to US inflation

emissions_target = False  # if True, will load historical data for a given emissions target
reduction_target = 0.7  # emissions target in proportion of baseline emissions

if fixed_carb_tax:
    print('Loading historical data for fixed carbon tax')
    if carb_cost <= 1.5 * 1e-4:
        dir_num = 9
    else:
        dir_num = 8
    dollar_adjustment = pd.read_csv('data/dollar_adjustment.csv', sep=';', decimal=',').set_index('year')

    if adjust:
        sol_all_adjusted = {}
        for y in range(1995, 2019):
            print(y)
            # print(dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost)
            sol_all_adjusted[y] = t.sol(y, dir_num, dollar_adjustment.loc[y]['dollar_adjusted'] * carb_cost)
        sol_all = sol_all_adjusted

    else:
        sol_all_unadjusted = {}
        for y in range(1995, 2019):
            print(y)
            sol_all_unadjusted[y] = t.sol(y, dir_num, carb_cost)
        sol_all = sol_all_unadjusted

if emissions_target:
    print('Loading historical data for emissions reduction target')
    dir_num = 9
    sol_all_target = {}
    carb_tax = {}
    for y in range(1995, 2019):
        print(y)
        year = str(y)
        path = 'results/' + year + '_' + str(dir_num)
        runs = pd.read_csv(path + '/runs')
        run = runs.iloc[np.argmin(np.abs(runs.emissions - runs.iloc[0].emissions * reduction_target))]
        print('Carbon tax =' + str(run.carb_cost * 1e6))
        carb_tax[y] = run.carb_cost
        sol_all_target[y] = t.sol(y, dir_num, carb_tax[y])
    sol_all = sol_all_target

if (not emissions_target) & (not fixed_carb_tax):
    print('No data to load, make a choice on what to load')

# %% compute traded quantities

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
co2_prod = {}

for y in sol_all.keys():
    print(y)
    # y = 2018
    co2_intensity = sol_all[y].co2_intensity.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector'])
    co2_intensity = co2_intensity.T.reset_index().drop(('index', ''), axis=1).T
    # Convert intensities in Mio T / $Mio
    co2_intensity = co2_intensity / 1e6
    co2_intensity.columns = ['co2_int']
    price_hat = sol_all[y].res.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector']).price_hat
    co2_prod[y] = sol_all[y].co2_prod.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'}).set_index(['row_country', 'row_sector'])
    co2_prod[y].columns = ['value_co2', 'new_co2']

    trade_em[y] = traded[y].groupby(level=[0,1,2]).sum().join(co2_intensity).join(price_hat).join(co2_prod[y])
    trade_em[y]['value'] = trade_em[y].value * trade_em[y].co2_int
    trade_em[y]['new'] = trade_em[y].new * trade_em[y].co2_int / trade_em[y].price_hat
    trade_em[y]['value_share'] = (trade_em[y].value / trade_em[y].value_co2)*100
    trade_em[y]['new_share'] = (trade_em[y].new / trade_em[y].new_co2)*100

#%% Calculate historical series to be plotted
years = [y for y in range(1995,2019)]
change_lv = []
change_pc = []
chshare_pp = []
chshare_pc = []

share_value = []
share_new = []

contribution = []

for y in range(1995,2019):
    print(y)
    change_lv.append(trade_em[y].sum().new - trade_em[y].sum().value)
    change_pc.append((trade_em[y].sum().new - trade_em[y].sum().value)/trade_em[y].sum().value*100)
    chshare_pp.append((trade_em[y].sum().new / co2_prod[y].sum().new_co2)*100 - (trade_em[y].sum().value / co2_prod[y].sum().value_co2)*100)
    chshare_pc.append(((trade_em[y].sum().new / co2_prod[y].sum().new_co2) - (trade_em[y].sum().value / co2_prod[y].sum().value_co2)) / (trade_em[y].sum().value / co2_prod[y].sum().value_co2) *100)
    share_value.append((trade_em[y].sum().value / co2_prod[y].sum().value_co2)*100)
    share_new.append((trade_em[y].sum().new / co2_prod[y].sum().new_co2) * 100)
    contribution.append((trade_em[y].sum().new - trade_em[y].sum().value) / (co2_prod[y].sum().new_co2 - co2_prod[y].sum().value_co2) * 100)

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
# ax.plot(years,share_new,label='Counterfactual',color = sns.color_palette()[2],lw=lw)

ax.set_ylabel('% of total emissions',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
# ax.legend()

plt.show()

#%%
print('Plotting historical contribution of reduction of traded emissions in total reduction of emissions %')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3

ax.plot(years,contribution,label='Baseline',color = sns.color_palette()[5],lw=lw)

ax.set_ylabel('% of total emissions reduction brought about by trade',fontsize = 20,color = sns.color_palette()[3])

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
# ax.legend()

plt.show()


# %% Calculate GSI for potential correlation checks
# %% compute distances for multiple quantities - function to compute distances

print('Computing distance indices for GSI comparison')


def compute_historic_distance(X):
    dist = []
    for i, y in enumerate(range(1995, 2019)):
        # print(y)
        dist.append(1 - pdist([X[i].value, X[i].new], metric='correlation'))
    return dist


years = [y for y in range(1995, 2019)]

all_flows_dist = compute_historic_distance([tot[y] for y in years])
country_to_country_dist = compute_historic_distance([tot[y].groupby(level=[0, 2]).sum() for y in years])
exports_dist = compute_historic_distance([tot[y].groupby(level=[0]).sum() for y in years])
imports_dist = compute_historic_distance([tot[y].groupby(level=[2]).sum() for y in years])
sectors = compute_historic_distance([tot[y].groupby(level=[1]).sum() for y in years])
c_s_c = compute_historic_distance([tot[y].groupby(level=[0, 1, 2]).sum() for y in years])
c_s = compute_historic_distance([tot[y].groupby(level=[0, 1]).sum() for y in years])

all_flows_dist_traded = compute_historic_distance([traded[y] for y in years])
country_to_country_dist_traded = compute_historic_distance([traded[y].groupby(level=[0, 2]).sum() for y in years])
exports_dist_traded = compute_historic_distance([traded[y].groupby(level=[0]).sum() for y in years])
imports_dist_traded = compute_historic_distance([traded[y].groupby(level=[2]).sum() for y in years])
sectors_traded = compute_historic_distance([traded[y].groupby(level=[1]).sum() for y in years])
c_s_c_traded = compute_historic_distance([traded[y].groupby(level=[0, 1, 2]).sum() for y in years])
c_s_traded = compute_historic_distance([traded[y].groupby(level=[0, 1]).sum() for y in years])

# all_flows_dist_local = compute_historic_distance([local[y] for y in years])
# country_to_country_dist_local = compute_historic_distance([local[y].groupby(level=[0,2]).sum() for y in years])
# exports_dist_local = compute_historic_distance([local[y].groupby(level=[0]).sum() for y in years])
# imports_dist_local = compute_historic_distance([local[y].groupby(level=[2]).sum() for y in years])
# sectors_local = compute_historic_distance([local[y].groupby(level=[1]).sum() for y in years])
# c_s_c_local = compute_historic_distance([local[y].groupby(level=[0,1,2]).sum() for y in years])
# c_s_local = compute_historic_distance([local[y].groupby(level=[0,1]).sum() for y in years])

gross_output_reduction_necessary = [-(tot[y].sum().new / tot[y].sum().value - 1) * 100 for y in years]
share_traded_change = [(traded[y].sum().new / tot[y].sum().new - traded[y].sum().value / tot[y].sum().value) * 100 for y
                       in years]
welfare_change = [
    -((sol_all[y].utility.new * sol_all[y].cons.groupby(level=2).sum().new).sum() / sol_all[y].cons.sum().new - 1) * 100
    for y in years]

#%% Correlating GSI and trade emissions saved

print('Plotting correlation between GSI and contribution of trade to emissions saved')

# gsi = c_s_c

fig, ax = plt.subplots(figsize=(12,8))

# ax1 = ax.twinx()

# ax.scatter([w for w in gsi], contribution,label='GSI',color = sns.color_palette()[3])
ax.scatter([w for w in country_to_country_dist_traded], contribution,color = sns.color_palette()[0],label='Country x Country index')
ax.scatter([w for w in sectors_traded], contribution,color = sns.color_palette()[1],label='Sector index')
# ax.scatter([w for w in c_s], contribution,color = sns.color_palette()[2],label='Country x Sector of origin')
ax.scatter([w for w in c_s_c_traded], contribution,label='GSI',color = sns.color_palette()[3])
# ax.scatter(imports_dist, contribution,label='imports_dist_traded',color = sns.color_palette()[5])
ax.legend()

# ax1.grid(visible=False)
# ax1.legend(loc = (-0.5,0))

ax.set_xlabel('GSI')
ax.set_ylabel('Trade contribution to emissions reduction (%)')
plt.title('Trade contribution to reduce emissions by  30%')

plt.show()

# %% Plot one GSI

color = sns.color_palette()[2]

fig, ax = plt.subplots(figsize=(12,8))

dist = c_s_c
one_bar = 0.99775

ax.plot(years,dist,lw=4,color=color)
ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
ax.set_title('Global Sustainability Index'
              ,color=color
              ,fontsize=28
              ,pad=15
              )
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0)

ax.hlines(y=one_bar,
           xmin=1995,
           xmax=2018,
           lw=3,
           ls = '--',
           color = color)

ax.annotate(1,
             xy=(1995, one_bar),
             xytext=(-50,-5),
            fontsize = 20,
             textcoords='offset points',color=color)

ax.annotate("Sustainable organization of trade\n with a $100 carbon tax",
            xy=(2009, one_bar), xycoords='data',
            xytext=(2009-2.5, one_bar-0.0004),
            textcoords='data',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color= 'black'),
            bbox=dict(boxstyle="round", fc="w")
            )

ax.grid(axis='x')

ax.set_yticks([0.9960, 0.9965, 0.9970, 0.9975, 0.99775])
ax.set_yticklabels(['0.9960', '0.9965', '0.9970', '0.9975', ''])

plt.tight_layout()

plt.show()