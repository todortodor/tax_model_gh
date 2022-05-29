# %% Import libraries
import os
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

import treatment_funcs_agri_ind_fe as t
import solver_funcs as s

# %% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

# %% Load needed data

# Load solution : Baseline = 2018, carbon cost = 100
print('Setting parameters for run')

y = 2018
year = str(y)
dir_num = 8
path = 'results/' + year + '_' + str(dir_num)
carb_cost = 1e-4

print('Loading ' + year + ' data, baseline for carbon tax ' + str(carb_cost * 1e6) + ' dollar per ton of CO2')
runs = pd.read_csv(path + '/runs')

# Baseline data
cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b = t.load_baseline(year)
sh = t.shares(cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b)

sector_list = iot_b.index.get_level_values(1).drop_duplicates().to_list()
S = len(sector_list)
country_list = iot_b.index.get_level_values(0).drop_duplicates().to_list()
C = len(country_list)

cons_traded_unit = cons_b.reset_index()[['row_country', 'row_sector', 'col_country']]
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit['row_country'] == cons_traded_unit['col_country'], ['value', 'new']] = 0
cons_traded_unit_np = cons_traded_unit.value.to_numpy()
iot_traded_unit = iot_b.reset_index()[['row_country', 'row_sector', 'col_country', 'col_sector']]
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit['row_country'] == iot_traded_unit['col_country'], ['value', 'new']] = 0
iot_traded_unit_np = iot_traded_unit.value.to_numpy()

sol_all = {}
if y not in sol_all.keys():
    print('Baseline year ' + str(y))
    sol_all[y] = t.sol(y, dir_num, carb_cost)

run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]

sigma = run.sigma
eta = run.eta
num = run.num

# Load population data
print('Loading labor data')

labor = pd.read_csv('data/World bank/labor_force/labor.csv')
labor.set_index('country', inplace=True)
labor.sort_index(inplace=True)
labor_year = labor[year]

#%% Traded flows

traded = {}
tot = {}
cons_trade = {}
iot_trade = {}

for y in sol_all.keys():
    print(y)

    cons = sol_all[y].cons
    iot = sol_all[y].iot

    cons_traded = cons.reset_index()
    cons_traded.loc[cons_traded['row_country'] == cons_traded['col_country'] , ['value','new']] = 0
    cons_traded = cons_traded.set_index(['row_country', 'row_sector', 'col_country'])
    cons_traded = pd.concat({'cons':cons_traded}, names=['col_sector']).reorder_levels([1,2,3,0])
    cons_trade[y] = cons_traded

    iot_traded = iot.reset_index()
    iot_traded.loc[iot_traded['row_country'] == iot_traded['col_country'] , ['value','new']] = 0
    iot_traded = iot_traded.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    iot_trade[y] = iot_traded

    tot[y] = pd.concat(
        [sol_all[y].iot, pd.concat({'cons': sol_all[y].cons}, names=['col_sector']).reorder_levels([1, 2, 3, 0])])
    temp = tot[y].reset_index()
    traded[y] = temp[temp['row_country'] != temp['col_country']].set_index(
        ['row_country', 'row_sector', 'col_country', 'col_sector'])
    del temp

#%% BASIC CHECKS

# %% Check - Main producer of a sector
# Conditions
sector = '01T02'        # Sector of consideration
variable = 'value'      # Change for 'new' to see counterfactual main producer

check_ps = sol_all[y].output.copy()
check_ps = check_ps.xs(sector,level=1).sort_values(variable, ascending=False)

#%% Check main exporter / importer of a sector
# Conditions
sector = '35'       # Sector of consideration
variable = 'value'  # Change for 'new' for counterfactual main exporter/importer
Exports = True      # If True main exporter, else main importer

if Exports:
    check_ts = traded[y].groupby(level=[0,1]).sum().copy()
    check_ts = check_ts.xs(sector, level=1).sort_values(variable, ascending=False)
else:
    check_ts = traded[y].groupby(level=[1,2]).sum().copy()
    check_ts = check_ts.xs(sector, level=0).sort_values(variable, ascending=False)

#%% REALLOCATION PLOTS

# %% Reallocation within sectors - Dataframe construction
print('Computing trade flows reallocation within sectors')

# Condition:
Total = True        # If true, reallocation across origin-destination pairs
Origin = False      # If true, reallocation across origins regardless of destination.
                    # If false, reallocation across destinations irrespective of origins

# Construct dataframe according to condition
sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

sector_change = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
    if Total:
        temp = traded[y].groupby(level=[0,1,2]).sum().xs(sector,level=1).new-traded[y].groupby(level=[0,1,2]).sum().xs(sector,level=1).value
    else:
        if Origin:
            temp = traded[y].groupby(level=[0,1]).sum().xs(sector,level=1).new-traded[y].groupby(level=[0,1]).sum().xs(sector,level=1).value
        else:
            temp = traded[y].groupby(level=[1,2]).sum().xs(sector,level=0).new-traded[y].groupby(level=[1,2]).sum().xs(sector,level=0).value

    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp>0].sum())
    sector_realloc_neg.append(temp[temp<0].sum())

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['traded'] = traded[y].groupby(level=1).sum().value.values
sector_dist_df['traded_new'] = traded[y].groupby(level=1).sum().new.values
sector_dist_df['realloc_pos'] = sector_realloc_pos
sector_dist_df['realloc_neg'] = sector_realloc_neg
sector_dist_df['change'] = sector_change

sector_dist_df['realloc_pos'] = np.abs(sector_dist_df['realloc_pos'])
sector_dist_df['realloc_neg'] = np.abs(sector_dist_df['realloc_neg'])
sector_dist_df['realloc'] = sector_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
sector_dist_df['realloc'] = sector_dist_df['realloc'] * np.sign(sector_dist_df['change'])
sector_dist_df['change_tot_nom'] = (sector_dist_df['change']+sector_dist_df['realloc'])
sector_dist_df['realloc_share_nom'] = (sector_dist_df['realloc']/sector_dist_df['change_tot_nom']) * np.sign(sector_dist_df['change'])

sector_dist_df['realloc_percent'] = (sector_dist_df['realloc']/sector_dist_df['traded'])*100
sector_dist_df['change_percent'] = (sector_dist_df['change']/sector_dist_df['traded'])*100
sector_dist_df['change_tot'] = (sector_dist_df['change_percent']+sector_dist_df['realloc_percent'])
sector_dist_df['realloc_share_neg'] = (sector_dist_df['realloc_percent']/sector_dist_df['change_tot']) * np.sign(sector_dist_df['change'])

# Calculate and print aggregate reallocation share
total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

if Total:
    print('Overall, '+str(total_output_reallocated_percent.round(2))+'% of traded volumes \nwould be reallocated within a sector \nacross country pairs for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
          )
else:
    if Origin:
        print('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of traded volumes \nwould be reallocated within a sector \nacross origins for a net reduction \nof trade flows of ' + str(
            total_output_decrease_percent.round(2)) + '%',
              )
    else:
        print('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of traded volumes \nwould be reallocated within a sector \nacross destinations for a net reduction \nof trade flows of ' + str(
            total_output_decrease_percent.round(2)) + '%',
              )

#%% Reallocation within sectors - Plot of nominal differences
print('Plotting trade reallocation within sectors in nominal differences')

sector_org = sector_dist_df[['industry', 'change', 'realloc','realloc_share_nom', 'change_tot_nom']].copy()
sector_pos = sector_org[sector_org['realloc_share_nom']>0].copy()
sector_pos.sort_values('change', ascending = True, inplace = True)
sector_neg1 = sector_org[sector_org['realloc_share_nom']<= -0.15].copy()
sector_neg1.sort_values('change',ascending = True, inplace = True)
sector_neg2 = sector_org[(sector_org['realloc_share_nom']> -0.15) & (sector_org['realloc_share_nom']<=0)].copy()
sector_neg2.sort_values('change',ascending = True, inplace = True)

sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(sector_use.industry
            ,sector_use.change/1e6
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of trade flows',
            # color=colors
            )

if Total:
    ax.bar(sector_use.industry
           , sector_use.realloc / 1e6
           , bottom=sector_use.change / 1e6
           , label='Reallocated trade flows across country pairs',
           # color=colors,
           hatch="////")
else:
    if Origin:
        ax.bar(sector_use.industry
                ,sector_use.realloc/1e6
                ,bottom = sector_use.change/1e6
                ,label='Reallocated trade flows across origins',
                # color=colors,
                hatch="////")
    else:
        ax.bar(sector_use.industry
               , sector_use.realloc / 1e6
               , bottom=sector_use.change / 1e6
               , label='Reallocated trade flows across destinations',
               # color=colors,
               hatch="////")

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)


ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Trillion $', fontsize = 20)

leg = ax.legend(fontsize=20,loc='lower right')

if Total:
    ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
        2)) + '% of traded volumes \nwould be reallocated within a sector \nacross country pairs for a net reduction \nof trade flows of ' + str(
        total_output_decrease_percent.round(2)) + '%',
                xy=(25, -0.28), fontsize=25, zorder=10, backgroundcolor='w')
else:
    if Origin:
        ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of traded volumes \nwould be reallocated within a sector \nacross origins for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
                 xy=(27,-0.28),fontsize=25,zorder=10,backgroundcolor='w')
    else:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of traded volumes \nwould be reallocated within a sector \nacross destinations for a net reduction \nof trade flows of ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(27, -0.28), fontsize=25, zorder=10, backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot_nom'].max()/1e6
min_lim = sector_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.1,max_lim+0.15)

plt.show()

#%% Reallocation within sectors - Plot changes in % of initial trade
print('Plotting trade reallocation within sectors in percentages')

sector_use = sector_dist_df.sort_values('change_percent', ascending=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change in trade volumes (%)',
            # color=colors
            )

if Total:
    ax.bar(sector_use.industry
           , sector_use.realloc_percent
           , bottom=sector_use.change_percent
           , label='Reallocated trade across country pairs (%)',
           # color=colors,
           hatch="////")
else:
    if Origin:
        ax.bar(sector_use.industry
                ,sector_use.realloc_percent
                ,bottom = sector_use.change_percent
                ,label='Reallocated trade across origins (%)',
                # color=colors,
                hatch="////")
    else:
        ax.bar(sector_use.industry
               , sector_use.realloc_percent
               , bottom=sector_use.change_percent
               , label='Reallocated trade across destinations (%)',
               # color=colors,
               hatch="////")

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial trade volumes', fontsize = 20)

leg = ax.legend(fontsize=20,loc='lower right')

if Total:
    ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
        2)) + '% of gross trade volumes \nwould be reallocated within a sector \nacross country pairs for a net reduction \nof trade flows of ' + str(
        total_output_decrease_percent.round(2)) + '%', xy=(25,-28),fontsize=25,zorder=10,backgroundcolor='w')
else:
    if Origin:
        ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross trade volumes \nwould be reallocated within a sector \nacross origins for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
                 xy=(27,-25),fontsize=25,zorder=10,backgroundcolor='w')
    else:
        ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross trade volumes \nwould be reallocated within a sector \nacross destinations for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
                 xy=(27,-25),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot'].max()
min_lim = sector_dist_df['change_tot'].min()
ax.set_ylim(min_lim-7,max_lim+10)

plt.show()

#%% Reallocation within countries - Dataframe construction
print('Computing trade reallocation within countries \nOrigin/Exporting country or Destination/Importing country')

# Conditions
Origin = False      # Determine if the country considered is the exporter or the importer
Total  = True       # If true, calculate total reallocation within an exporter/importer across destinations/origins and sectors
Sector = False      # If true, calculate reallocation within an exporter/importer across sectors.
                    # If false reallocation is across destinations/origins irrespective of traded sectors.

# Dataframe construction
country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_change = []
country_realloc_pos = []
country_realloc_neg = []

if Origin:
    for country in country_list:
        # print(country)
        if Total:
            temp = traded[y].groupby(level=[0, 1, 2]).sum().xs(country, level=0).new - traded[y].groupby(
                level=[0, 1, 2]).sum().xs(country, level=0).value
        else:
            if Sector:
                temp = traded[y].groupby(level=[0,1]).sum().xs(country,level=0).new-traded[y].groupby(level=[0,1]).sum().xs(country,level=0).value

            else:
                temp = traded[y].groupby(level=[0,2]).sum().xs(country, level=0).new - traded[y].groupby(level=[0,2]).sum().xs(country, level=0).value

        country_change.append(temp.sum())
        country_realloc_pos.append(temp[temp > 0].sum())
        country_realloc_neg.append(temp[temp < 0].sum())
        country_dist_df = pd.DataFrame(index=country_list)
        country_dist_df['traded'] = traded[y].groupby(level=0).sum().value.values
        country_dist_df['traded_new'] = traded[y].groupby(level=0).sum().new.values

else:
    for country in country_list:
        if Total:
            temp = traded[y].groupby(level=[0, 1, 2]).sum().xs(country, level=2).new - traded[y].groupby(
                level=[0, 1, 2]).sum().xs(country, level=2).value
        else:
            if Sector:
                temp = traded[y].groupby(level=[1,2]).sum().xs(country, level=1).new - traded[y].groupby(level=[1,2]).sum().xs(country, level=1).value
            else:
                temp = traded[y].groupby(level=[0, 2]).sum().xs(country, level=0).new - traded[y].groupby(
                    level=[0, 2]).sum().xs(country, level=0).value

        country_change.append(temp.sum())
        country_realloc_pos.append(temp[temp > 0].sum())
        country_realloc_neg.append(temp[temp < 0].sum())
        country_dist_df = pd.DataFrame(index=country_list)
        country_dist_df['traded'] = traded[y].groupby(level=2).sum().value.values
        country_dist_df['traded_new'] = traded[y].groupby(level=2).sum().new.values

country_dist_df['realloc_pos'] = country_realloc_pos
country_dist_df['realloc_neg'] = country_realloc_neg
country_dist_df['change'] = country_change
country_dist_df['share_percent'] = (country_dist_df['traded']/country_dist_df['traded'].sum())*100
country_dist_df['share_new_percent'] = (country_dist_df['traded_new']/country_dist_df['traded_new'].sum())*100

country_dist_df['realloc_pos'] = np.abs(country_dist_df['realloc_pos'])
country_dist_df['realloc_neg'] = np.abs(country_dist_df['realloc_neg'])
country_dist_df['realloc'] = country_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
country_dist_df['realloc'] = country_dist_df['realloc'] * np.sign(country_dist_df['change'])
country_dist_df['change_tot_nom'] = (country_dist_df['change']+country_dist_df['realloc'])

country_dist_df['realloc_percent'] = (country_dist_df['realloc']/country_dist_df['traded'])*100
country_dist_df['change_percent'] = (country_dist_df['change']/country_dist_df['traded'])*100
country_dist_df['total_change'] = country_dist_df['realloc_percent'] + country_dist_df['change_percent']

# Calculate and print aggregate reallocation share
total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

if Origin:
    if Total:
        print('Overall, '+str(total_output_reallocated_percent.round(2))+'% of trade volumes \nwould be reallocated within an \nexporting country across sectors and destinations \nfor a net reduction of trade flows \nof '+str(total_output_decrease_percent.round(2))+'%')
    else:
        if Sector:
            print('Overall, '+str(total_output_reallocated_percent.round(2))+'% of trade volumes \nwould be reallocated within an \nexporting country across sectors \nfor a net reduction of trade flows \nof '+str(total_output_decrease_percent.round(2))+'%')
        else:
            print('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across destinations \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%')
else:
    if Total:
        print('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins and sectors \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%')
    else:
        if Sector:
            print('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across sectors \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%')
        else:
            print('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%')

#%% Within country reallocation - Plot of nominal differences
print('Plotting trade reallocation within countries in nominal differences')

country_dist_df.sort_values('change_tot_nom',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.change/1e6
                # ,bottom = country_dist_df.realloc_neg
                ,label='Net change in exports',
                # color=colors
                )
    if Total:
        ax.bar(country_dist_df.index.get_level_values(0)
               , country_dist_df.realloc / 1e6
               , bottom=country_dist_df.change / 1e6
               , label='Reallocated exports across sectors and destinations',
               # color=colors,
               hatch="////")
    else:
        if Sector:
            ax.bar(country_dist_df.index.get_level_values(0)
                    ,country_dist_df.realloc/1e6
                    ,bottom = country_dist_df.change/1e6
                    ,label='Reallocated exports across sectors',
                    # color=colors,
                    hatch="////")
        else:
            ax.bar(country_dist_df.index.get_level_values(0)
                    ,country_dist_df.realloc/1e6
                    ,bottom = country_dist_df.change/1e6
                    ,label='Reallocated exports across destinations',
                    # color=colors,
                    hatch="////")
else:
    ax.bar(country_dist_df.index.get_level_values(0)
           , country_dist_df.change / 1e6
           # ,bottom = country_dist_df.realloc_neg
           , label='Net change in imports',
           # color=colors
           )

    if Total:
        ax.bar(country_dist_df.index.get_level_values(0)
               , country_dist_df.realloc / 1e6
               , bottom=country_dist_df.change / 1e6
               , label='Reallocated imports across origins and sectors',
               # color=colors,
               hatch="////")
    else:
        if Sector:
            ax.bar(country_dist_df.index.get_level_values(0)
                   , country_dist_df.realloc / 1e6
                   , bottom=country_dist_df.change / 1e6
                   , label='Reallocated imports across sectors',
                   # color=colors,
                   hatch="////")
        else:
            ax.bar(country_dist_df.index.get_level_values(0)
                   , country_dist_df.realloc / 1e6
                   , bottom=country_dist_df.change / 1e6
                   , label='Reallocated imports across origins',
                   # color=colors,
                   hatch="////")

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Trillion $',fontsize = 20)

leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)


if Origin:
    if Total:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across sectors and destinations \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(41, -0.2), fontsize=25, zorder=10, backgroundcolor='w')
    else:
        if Sector:
            ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of trade volumes \nwould be reallocated within an \nexporting country across sectors \nfor a net reduction of trade flows \nof '+str(total_output_decrease_percent.round(2))+'%',
                 xy=(41,-0.2),fontsize=25,zorder=10,backgroundcolor='w')
        else:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across destinations \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -0.12), fontsize=25, zorder=10, backgroundcolor='w')
else:
    if Total:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins and sectors \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(41, -0.15), fontsize=25, zorder=10, backgroundcolor='w')
    else:
        if Sector:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across sectors \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(41, -0.15), fontsize=25, zorder=10, backgroundcolor='w')
        else:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -0.12), fontsize=25, zorder=10, backgroundcolor='w')

max_lim = country_dist_df['change_tot_nom'].max()/1e6
min_lim = country_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.02,max_lim+0.02)

plt.show()

#%% Reallocation within countries - Plot changes in % of initial trade
print('Plotting trade reallocation within sectors in percentages')

country_dist_df.sort_values('change_percent',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.change_percent
                # ,bottom = country_dist_df.realloc_neg
                ,label='Net change in exports (%)',
                # color=colors
                )

    if Total:
        ax.bar(country_dist_df.index.get_level_values(0)
               , country_dist_df.realloc_percent
               , bottom=country_dist_df.change_percent
               , label='Reallocated exports across sectors and destinations (%)',
               # color=colors,
               hatch="////")
    else:
        if Sector:
            ax.bar(country_dist_df.index.get_level_values(0)
                    ,country_dist_df.realloc_percent
                    ,bottom = country_dist_df.change_percent
                    ,label='Reallocated exports across sectors (%)',
                    # color=colors,
                    hatch="////")
        else:
            ax.bar(country_dist_df.index.get_level_values(0)
                   , country_dist_df.realloc_percent
                   , bottom=country_dist_df.change_percent
                   , label='Reallocated exports across destinations (%)',
                   # color=colors,
                   hatch="////")
else:
    ax.bar(country_dist_df.index.get_level_values(0)
           , country_dist_df.change_percent
           # ,bottom = country_dist_df.realloc_neg
           , label='Net change in imports (%)',
           # color=colors
           )

    if Total:
        ax.bar(country_dist_df.index.get_level_values(0)
               , country_dist_df.realloc_percent
               , bottom=country_dist_df.change_percent
               , label='Reallocated imports across origins and sectors (%)',
               # color=colors,
               hatch="////")
    else:
        if Sector:
            ax.bar(country_dist_df.index.get_level_values(0)
               , country_dist_df.realloc_percent
               , bottom=country_dist_df.change_percent
               , label='Reallocated imports across sectors (%)',
               # color=colors,
               hatch="////")
        else:
            ax.bar(country_dist_df.index.get_level_values(0)
                   , country_dist_df.realloc_percent
                   , bottom=country_dist_df.change_percent
                   , label='Reallocated imports across origins (%)',
                   # color=colors,
                   hatch="////")

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

if Origin:
    ax.set_ylabel('% of initial exports',
              fontsize = 20)
else:
    ax.set_ylabel('% of initial imports',
                  fontsize=20)

leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

if Origin:
    if Total:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across sectors \nand destinations for a net \nreduction of trade flows of ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(43, -25), fontsize=25, zorder=10, backgroundcolor='w')
    else:
        if Sector:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across sectors \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -20), fontsize=25, zorder=10, backgroundcolor='w')
        else:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across destinations \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -20), fontsize=25, zorder=10, backgroundcolor='w')
else:
    if Total:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins \nand sectors for a net \nreduction of trade flows of ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(43, -15), fontsize=25, zorder=10, backgroundcolor='w')
    else:
        if Sector:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across sectors \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -20), fontsize=25, zorder=10, backgroundcolor='w')
        else:
            ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
                2)) + '% of trade volumes \nwould be reallocated within an \nimporting country across origins \nfor a net reduction of trade flows \nof ' + str(
                total_output_decrease_percent.round(2)) + '%',
                        xy=(41, -20), fontsize=25, zorder=10, backgroundcolor='w')

max_lim = country_dist_df['total_change'].max()
min_lim = country_dist_df['total_change'].min()
ax.set_ylim(min_lim-3,max_lim+3)

plt.show()

# %% CHANGES IN TRADE SHARES

#%% Changes in sectoral share of traded output - What sector should we trade relatively more?
# Construct dataframe
print('Computing share of output traded sector-wise')
trade_sh = traded[y].groupby(level=[1]).sum()
trade_sh = trade_sh.div(tot[y].groupby(level=[1]).sum())*100

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['value'] = trade_sh.value.values
sector_dist_df['new'] = trade_sh.new.values
sector_dist_df['diff'] = sector_dist_df.new - sector_dist_df.value
sector_dist_df['diff_pc'] = sector_dist_df['diff'] /sector_dist_df['value']*100

#%% Changes in sectoral share of traded ouput - difference in percentage points
print('Plotting sectoral share of output traded changes by sector in pp')

sector_use = sector_dist_df.sort_values('diff', ascending=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(sector_use.industry
            ,sector_use['diff']
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Change in share of output traded',
            # color=colors
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percentage points', fontsize = 20)

leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['diff'].max()
min_lim = sector_dist_df['diff'].min()
ax.set_ylim(min_lim-1.3,max_lim+1.8)

plt.show()

#%% Changes in sectoral share of traded ouput - difference in % of initial share
print('Plotting change in sectoral share of output traded in percentages')

sector_use = sector_dist_df.sort_values('diff_pc', ascending=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)


ax.bar(sector_use.industry
            ,sector_use['diff_pc']
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Change in share of output traded (%)',
            # color=colors
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial share', fontsize = 20)

leg = ax.legend(fontsize=20,loc='upper left')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['diff_pc'].max()
min_lim = sector_dist_df['diff_pc'].min()
ax.set_ylim(min_lim-6,max_lim+5)

plt.show()

#%% Changes in geographical share of traded output - What country should export/import relatively more?
print('Computing trade reallocation country-wise - Origin/Exporting country or Destination/Importing country')

#Condition
Origin = False       # If true we consider export share of total output
                    # If False we consider import share of total demand (differ from output by the deficit)

# Construct dataframe
if Origin:
    trade_sh = traded[y].groupby(level=[0]).sum()
    trade_sh = trade_sh.div(tot[y].groupby(level=[0]).sum()) * 100
else:
    trade_sh = traded[y].groupby(level=[2]).sum()
    trade_sh = trade_sh.div(tot[y].groupby(level=[2]).sum()) * 100

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_dist_df = pd.DataFrame(index=country_list)
country_dist_df['value'] = trade_sh.value.values
country_dist_df['new'] = trade_sh.new.values
country_dist_df['diff'] = country_dist_df.new - country_dist_df.value
country_dist_df['diff_pc'] = country_dist_df['diff'] /country_dist_df['value']*100

#%% Changes in geographical share of traded output - difference in percentage points
print('Plotting change in export/import share in pp')

country_dist_df.sort_values('diff',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df['diff']
                # ,bottom = country_dist_df.realloc_neg
                ,label='Change in export share',
                # color=colors
                )
else:
    ax.bar(country_dist_df.index.get_level_values(0)
           , country_dist_df['diff']
           # ,bottom = country_dist_df.realloc_neg
           , label='Change in import share',
           # color=colors
           )

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percentage points',
              fontsize = 20)

leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

max_lim = country_dist_df['diff'].max()
min_lim = country_dist_df['diff'].min()
ax.set_ylim(min_lim-0.2,max_lim+0.2)

plt.show()

#%% Changes in geographical share of traded output - difference in % of initial share
print('Plotting change in import/export share in percentages')

country_dist_df.sort_values('diff_pc',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.diff_pc
            # ,bottom = country_dist_df.realloc_neg
            ,label='% of initial share',
            # color=colors
            )

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)

ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

if Origin:
    ax.set_ylabel('% of initial export share',
              fontsize = 20)
else:
    ax.set_ylabel('% of initial import share',
                  fontsize=20)

leg = ax.legend(fontsize=20,loc='upper left')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

max_lim = country_dist_df['diff_pc'].max()
min_lim = country_dist_df['diff_pc'].min()
ax.set_ylim(min_lim-1,max_lim+1)

plt.show()

# %% COMPOSITION EFFECTS

# %% Changes in sectoral composition of a country's production
print('Plotting gross output composition for '+country)

# Conditions
country = 'SAU'             # Country of consideration
variable = 'change_share'   # 'value_share' for current composition
                            # 'new_share' for counterfactual composition
                            # 'change_share' for relative change in share

# Construct sectoral output shares
sc_df = sol_all[y].output.copy()
sc_df['value_total'] = sc_df.groupby(level=0).value.transform('sum')
sc_df['new_total'] = sc_df.groupby(level=0).new.transform('sum')
sc_df['value_share'] = (sc_df.value / sc_df.value_total)*100
sc_df['new_share'] = (sc_df.new / sc_df.new_total)*100
sc_df['change_share'] = (sc_df.new_share / sc_df.value_share - 1)*100

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)

# Plot
data = sc_df.xs(country, level=0)[variable].to_list()
indicators = sector_map.group_code.to_list()
group_labels = sector_map.group_label.to_list()

indicators_sorted , sector_list_full_sorted , data_sorted , group_labels_sorted  = zip(*sorted(zip(indicators, sector_list_full, data , group_labels)))
group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

fig, ax = plt.subplots(figsize=(18,10))
color = sns.color_palette()[7]
palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
colors = [palette[ind-1] for ind in indicators_sorted]

ax.bar(sector_list_full_sorted, data_sorted, color=colors,width=0.5)
ax.set_xticklabels(sector_list_full_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.tick_params(axis='x', which='major', pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax.legend(handles=handles,fontsize=20)

plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
plt.suptitle('Sectoral output composition in country '+country+' (%)',size = 30,y=0.96)

plt.tight_layout()

plt.show()

# %% Changes in sectoral composition of a country's exports OR imports
# Conditions
country = 'BRA'                 # Country of interest
Exports = True                  # If true we're looking at the sectoral composition of exports
                                # If false we're looking at the sectoral composition of imports
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting sectoral composition of exports for country '+country)
else:
    print('Plotting sectoral composition of imports for country ' + country)

# Construct sectoral output shares
if Exports:
    sc_df = traded[y].groupby(level=[0,1]).sum().copy()
    sc_df['value_total'] = sc_df.groupby(level=0).value.transform('sum')
    sc_df['new_total'] = sc_df.groupby(level=0).new.transform('sum')
else:
    sc_df = traded[y].groupby(level=[1,2]).sum().copy()
    sc_df['value_total'] = sc_df.groupby(level=1).value.transform('sum')
    sc_df['new_total'] = sc_df.groupby(level=1).new.transform('sum')

sc_df['value_share'] = (sc_df.value / sc_df.value_total)*100
sc_df['new_share'] = (sc_df.new / sc_df.new_total)*100
sc_df['change_share'] = (sc_df.new_share / sc_df.value_share - 1)*100
sc_df['change_value'] = (sc_df.new / sc_df.value - 1)*100

# Plot
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)

if Exports:
    data1 = sc_df.xs(country, level=0)[variable1].to_list()
    data2 = sc_df.xs(country, level=0)[variable2].to_list()
else:
    data1 = sc_df.xs(country, level=1)[variable1].to_list()
    data2 = sc_df.xs(country, level=1)[variable2].to_list()

indicators = sector_map.group_code.to_list()
group_labels = sector_map.group_label.to_list()
indicators_sorted , sector_list_full_sorted , data1_sorted, data2_sorted , group_labels_sorted  = zip(*sorted(zip(indicators, sector_list_full, data1 , data2, group_labels)))
group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
colors = [palette[ind-1] for ind in indicators_sorted]

fig, ax = plt.subplots(2,1,figsize=(18,14))
# Subplot 1 - Initial sectoral composition
ax[0].bar(sector_list_full_sorted, data1_sorted, color=colors,width=0.5)
ax[0].set_xticklabels(sector_list_full_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax[0].tick_params(axis='x', which='major', pad=-9)
ax[0].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[0].set_ylabel('Share in total exports (%)', fontsize=20)
else:
    ax[0].set_ylabel('Share in total imports (%)', fontsize=20)

ax[0].margins(x=0.01)

handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax[0].legend(handles=handles,fontsize=20)

# Subplot 2 - Change in export share of a given sector in %
ax[1].bar(sector_list_full_sorted, data2_sorted, color=colors,width=0.5)
ax[1].set_xticklabels(sector_list_full_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax[1].tick_params(axis='x', which='major', pad=-9)
ax[1].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[1].set_ylabel('Change in export share (%)', fontsize=20)
else:
    ax[1].set_ylabel('Change in import share (%)', fontsize=20)

ax[1].margins(x=0.01)

# plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
if Exports:
    plt.suptitle('Sectoral exports composition in country '+country+' (% and % change)',size = 30,y=0.96)
else:
    plt.suptitle('Sectoral imports composition in country '+country+' (% and % change)',size = 30,y=0.96)

plt.tight_layout()

plt.show()

# %% Changes in geographic composition of a country's exports OR imports
# Conditions
country = 'IRL'                 # Country of interest
Exports = False                 # If true, we're looking at the distribution of destinations conditional on considering a country's exports
                                # If false, we're looking at the distribution of origins within a given country's imports
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting geographic distribution of destinations for '+country+"'s exports")
else:
    print('Plotting geographic distribution of origins for '+country+"'s imports")

# Construct geographic shares
cc_df = traded[y].groupby(level=[0,2]).sum().copy()

if Exports:
    cc_df['value_total'] = cc_df.groupby(level=0).value.transform('sum')
    cc_df['new_total'] = cc_df.groupby(level=0).new.transform('sum')
else:
    cc_df['value_total'] = cc_df.groupby(level=1).value.transform('sum')
    cc_df['new_total'] = cc_df.groupby(level=1).new.transform('sum')

cc_df['value_share'] = (cc_df.value / cc_df.value_total)*100
cc_df['new_share'] = (cc_df.new / cc_df.new_total)*100
cc_df['change_share'] = (cc_df.new_share / cc_df.value_share - 1)*100
cc_df['change_value'] = (cc_df.new / cc_df.value - 1)*100

# Plot
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')

if Exports:
    data_df = cc_df.xs(country, level=0).rename_axis('country').join(country_map)
else:
    data_df = cc_df.xs(country, level=1).rename_axis('country').join(country_map)

data_df.loc['TWN', 'Continent'] = 'Asia'
data_df.loc['ROW', 'Continent'] = 'Africa'
data_df.loc['AUS', 'Continent'] = 'Asia'
data_df.loc['NZL', 'Continent'] = 'Asia'
data_df.loc['CRI', 'Continent'] = 'South America'
data_df.loc['RUS', 'Continent'] = 'Asia'
data_df.loc['SAU', 'Continent'] = 'Africa'
data_df.loc[data_df.Continent == 'South America' , 'group_code'] = 1
data_df.loc[data_df.Continent == 'Asia' , 'group_code'] = 2
data_df.loc[data_df.Continent == 'Europe' , 'group_code'] = 3
data_df.loc[data_df.Continent == 'North America' , 'group_code'] = 4
data_df.loc[data_df.Continent == 'Africa' , 'group_code'] = 5
data1 = data_df[variable1].to_list()
data2 = data_df[variable2].to_list()


indicators = data_df.group_code.to_list()
group_labels = data_df.Continent.to_list()
country_t_list = data_df.index.values.tolist()
indicators_sorted , country_t_list_sorted , data1_sorted, data2_sorted , group_labels_sorted  = zip(*sorted(zip(indicators, country_t_list, data1, data2 , group_labels)))
group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }

colors = [continent_colors[data_df.loc[cou,'Continent']] for cou in country_t_list_sorted]
# colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(2,1, figsize=(18,14))

# Subplot 1 - Initial distribution of destinations / origins
ax[0].bar(country_t_list_sorted, data1_sorted, color=colors,width=0.5)
ax[0].set_xticklabels(country_t_list_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=18)
ax[0].tick_params(axis='x', which='major', pad=-9)
ax[0].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[0].set_ylabel('Share of total exports (%)', fontsize=20)
else:
    ax[0].set_ylabel('Share of total imports (%)', fontsize=20)
ax[0].margins(x=0.01)

handles = [mpatches.Patch(color=palette[cou], label=group_labels_sorted[cou]) for cou,group in enumerate(group_labels_sorted)]
ax[0].legend(handles=handles,fontsize=20)

# Subplot 2 - Change in a destination / origin share in %
ax[1].bar(country_t_list_sorted, data2_sorted, color=colors,width=0.5)
ax[1].set_xticklabels(country_t_list_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=18)
ax[1].tick_params(axis='x', which='major', pad=-9)
ax[1].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[1].set_ylabel('Change in export share (%)', fontsize=20)
else:
    ax[1].set_ylabel('Change in import share (%)', fontsize=20)
ax[1].margins(x=0.01)


# plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
if Exports:
    plt.suptitle('Geographic distribution of '+country+"'s exports",size = 30,y=0.96)
else:
    plt.suptitle('Geographic distribution of '+country+"'s imports",size = 30,y=0.96)

plt.tight_layout()

plt.show()

# %% Changes in geographic distribution of sectoral exports OR imports
# Conditions
sector = '24'                   # Sector of consideration
Exports = False                 # If true, we're looking at the geographic distribution of who is exporting this sector
                                # If false, we're looking at the geographic distribution of who is importing this sector
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting geographic distribution of origins of trade for sector '+sector)
else:
    print('Plotting geographic distribution of destinations of trade for sector' +sector)

# Construct geographic shares
if Exports:
    ss_df = traded[y].groupby(level=[0, 1]).sum().copy()
    ss_df['value_total'] = ss_df.groupby(level=1).value.transform('sum')
    ss_df['new_total'] = ss_df.groupby(level=1).new.transform('sum')
else:
    ss_df = traded[y].groupby(level=[1,2]).sum().copy()
    ss_df['value_total'] = ss_df.groupby(level=0).value.transform('sum')
    ss_df['new_total'] = ss_df.groupby(level=0).new.transform('sum')

ss_df['value_share'] = (ss_df.value / ss_df.value_total)*100
ss_df['new_share'] = (ss_df.new / ss_df.new_total)*100
ss_df['change_share'] = (ss_df.new_share / ss_df.value_share - 1)*100
ss_df['change_value'] = (ss_df.new / ss_df.value - 1)*100

# Plot
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')

if Exports:
    data_df = ss_df.xs(sector, level=1).rename_axis('country').join(country_map)
else:
    data_df = ss_df.xs(sector, level=0).rename_axis('country').join(country_map)

data_df.loc['TWN', 'Continent'] = 'Asia'
data_df.loc['ROW', 'Continent'] = 'Africa'
data_df.loc['AUS', 'Continent'] = 'Asia'
data_df.loc['NZL', 'Continent'] = 'Asia'
data_df.loc['CRI', 'Continent'] = 'South America'
data_df.loc['RUS', 'Continent'] = 'Asia'
data_df.loc['SAU', 'Continent'] = 'Africa'
data_df.loc[data_df.Continent == 'South America' , 'group_code'] = 1
data_df.loc[data_df.Continent == 'Asia' , 'group_code'] = 2
data_df.loc[data_df.Continent == 'Europe' , 'group_code'] = 3
data_df.loc[data_df.Continent == 'North America' , 'group_code'] = 4
data_df.loc[data_df.Continent == 'Africa' , 'group_code'] = 5
data1 = data_df[variable1].to_list()
data2 = data_df[variable2].to_list()

indicators = data_df.group_code.to_list()
group_labels = data_df.Continent.to_list()
country_t_list = data_df.index.values.tolist()
indicators_sorted , country_t_list_sorted , data1_sorted, data2_sorted , group_labels_sorted  = zip(*sorted(zip(indicators, country_t_list, data1, data2 , group_labels)))
group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }

colors = [continent_colors[data_df.loc[cou,'Continent']] for cou in country_t_list_sorted]
# colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(2,1, figsize=(18,14))

# Subplot 1 - Initial geographic distribution of origins/destinations
ax[0].bar(country_t_list_sorted, data1_sorted, color=colors,width=0.5)
ax[0].set_xticklabels(country_t_list_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=18)
ax[0].tick_params(axis='x', which='major', pad=-9)
ax[0].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[0].set_ylabel('Share of total exports (%)', fontsize=20)
else:
    ax[0].set_ylabel('Share of total imports (%)', fontsize=20)
ax[0].margins(x=0.01)

# Subplot 2 - Change in a destination / origin share in %
ax[1].bar(country_t_list_sorted, data2_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
ax[1].set_xticklabels(country_t_list_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=18)
ax[1].tick_params(axis='x', which='major', pad=-9)
ax[1].tick_params(axis='y', labelsize = 20)
if Exports:
    ax[1].set_ylabel('Change in export share (%)', fontsize=20)
else:
    ax[1].set_ylabel('Change in import share (%)', fontsize=20)
ax[1].margins(x=0.01)

# handles = []
# for ind in indicators_sorted:
handles = [mpatches.Patch(color=palette[cou], label=group_labels_sorted[cou]) for cou,group in enumerate(group_labels_sorted)]
ax[0].legend(handles=handles,fontsize=20)


# ax.legend(group_labels_sorted)

# plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
# if Exports:
#     plt.suptitle('Geographic composition of export destinations in country '+country+' (%)',size = 30,y=0.96)
# else:
#     plt.suptitle('Geographic composition of import origins in country '+country+' (%)',size = 30,y=0.96)

plt.tight_layout()

plt.show()