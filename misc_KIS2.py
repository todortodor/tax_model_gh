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

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

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
    # y = 2018

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

    # traded[y] = pd.concat([iot_traded,cons_traded])
    tot[y] = pd.concat(
        [sol_all[y].iot, pd.concat({'cons': sol_all[y].cons}, names=['col_sector']).reorder_levels([1, 2, 3, 0])])
    temp = tot[y].reset_index()
    traded[y] = temp[temp['row_country'] != temp['col_country']].set_index(
        ['row_country', 'row_sector', 'col_country', 'col_sector'])
    del temp

# %% Construct bilateral trade flows weighted by output and imports
# Trade in final goods / intermediate inputs / total

cons_imported = sol_all[y].cons.groupby(level=[1, 2]).sum().reset_index().copy()
iot_imported = sol_all[y].iot.groupby(level=[1, 2]).sum().reset_index().copy()
imports = tot[y].groupby(level=[1,2]).sum().reset_index().copy()
output = sol_all[y].output.reset_index().rename(columns={'country': 'row_country', 'sector': 'row_sector'})

# Trade in final goods
cons_flows = pd.merge(
    cons_trade[y].droplevel(3, axis=0).reset_index().copy(),
    output,
    'left',
    on=['row_country', 'row_sector'],
    suffixes=('_traded', '_output')
)
cons_flows = pd.merge(
    cons_flows,
    cons_imported,
    'left',
    on=['row_sector', 'col_country']
)
cons_flows.rename(columns={'value': 'value_imports', 'new': 'new_imports'}, inplace=True)

# Trade in intermediate inputs
iot_flows = pd.merge(
    iot_trade[y].groupby(level=[0, 1, 2]).sum().reset_index().copy(),
    output,
    'left',
    on=['row_country', 'row_sector'],
    suffixes=('_traded', '_output')
)
iot_flows = pd.merge(
    iot_flows,
    iot_imported,
    'left',
    on=['row_sector', 'col_country']
)
iot_flows.rename(columns={'value': 'value_imports', 'new': 'new_imports'}, inplace=True)


# Total trade
trade_flows = pd.merge(
    traded[y].groupby(level=[0, 1, 2]).sum().reset_index().copy(),
    output,
    'left',
    on=['row_country', 'row_sector'],
    suffixes=('_traded', '_output')
)
trade_flows = pd.merge(
    trade_flows,
    imports,
    'left',
    on=['row_sector', 'col_country']
)
trade_flows.rename(columns={'value': 'value_imports', 'new': 'new_imports'}, inplace=True)

# %% Load country and sector infos
world = pd.read_csv('data/countries_codes_and_coordinates.csv')[['Alpha-3 code','Latitude (average)', 'Longitude (average)']]
world.columns = ['country','latitude','longitude']
for column in world.columns:
    world[column] = world[column].str.replace('"','')
    world[column] = world[column].str.replace(' ','')
    # world[column] = world[column].astype('str')
world['country'] = world['country'].astype('string')
world.drop_duplicates('country',inplace=True)
# world['country'] = world['country'].to_string()
world['latitude'] = pd.to_numeric(world['latitude'])
world['longitude'] = pd.to_numeric(world['longitude'])
# world.set_index('country',inplace=True)

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_map['row_sector'] = sector_list
sector_map = sector_map.reset_index().set_index('row_sector')

country_map = pd.read_csv('data/country_continent.csv',sep=';')

country_map['region_1'] = country_map.region_1.str.split().apply(reversed).apply(' '.join)

# %% Format dataframe

def network_create(data):
    c_c = data.copy()

    c_c['row_sector'] = c_c.row_sector.replace(sector_map['industry'])
    c_c = c_c.groupby(['row_country','row_sector','col_country']).sum()
    c_c = c_c.rename_axis(['Exporter','Sector','Importer'])

    c_c.columns = ['value_traded', 'new_traded', 'Production (Mio$)', 'new_output',
                   'value_imports', 'new_imports']

    # c_c['value'] = c_c['value']*1e6
    # c_c['new'] = c_c['new']*1e6
    # c_c.columns = ['value_traded', 'new_traded', 'Production (Mio$)', 'new_output',
    #                'value_imports', 'new_imports',
    #                'Trade flow (1/$)','New trade flow (1/$)', 'Trade flow change (%)']
    # c_c['Trade flow change (1/$)'] = c_c['New trade flow (1/$)'] - c_c['Trade flow (1/$)']
    #
    # c_c = c_c.reset_index().pivot(index=['Exporter','Importer'],columns='Sector', values = ['value_traded', 'new_traded', 'Production (Mio$)', 'new_output',
    #                'value_imports', 'new_imports',
    #                'Trade flow (1/$)','New trade flow (1/$)', 'Trade flow change (%)', 'Trade flow change (1/$)'])
    # c_c = pd.concat({' Total':c_c.groupby(axis=1,level=0).sum()},axis=1).reorder_levels([1,0],axis=1).join(c_c)

    c_c = c_c.reset_index().pivot(index=['Exporter','Importer'],columns='Sector', values = ['value_traded', 'new_traded', 'Production (Mio$)', 'new_output', 'value_imports', 'new_imports'])
    c_c = pd.concat({' Total':c_c.groupby(axis=1,level=0).sum()},axis=1).reorder_levels([1,0],axis=1).join(c_c)

    # c_c['value'] = c_c['value_traded'] / (c_c['value_output'] * c_c['value_imports'])
    # c_c['new'] = c_c['new_traded'] / (c_c['new_output'] * c_c['new_imports'])
    # c_c['change'] = (c_c['new'] / c_c['value'] -1)*100

    c_c[pd.MultiIndex.from_product([['Trade flow weighted (1/$)'],c_c.columns.get_level_values(1).drop_duplicates()])] = c_c['value_traded'] / (c_c['Production (Mio$)'] * c_c['value_imports'])*1e6
    c_c[pd.MultiIndex.from_product([['New Trade flow weighted (1/$)'],c_c.columns.get_level_values(1).drop_duplicates()])] = c_c['new_traded'] / (c_c['new_output'] * c_c['new_imports'])*1e6
    c_c[pd.MultiIndex.from_product([['Trade flow weighted change (%)'],c_c.columns.get_level_values(1).drop_duplicates()])] = (c_c['New Trade flow weighted (1/$)']/c_c['Trade flow weighted (1/$)']- 1)*100
    c_c = c_c.drop(['value_traded', 'new_traded', 'new_output', 'value_imports', 'new_imports', 'New Trade flow weighted (1/$)'],axis=1)
    c_c = c_c.reorder_levels([1,0],axis=1).sort_index(axis=1)
    c_c.rename_axis(['Sector','Quantity'],axis=1,inplace=True)

    for c in c_c.index.get_level_values(0).drop_duplicates():
        if c == 'TWN':
            c_c.loc[c,'Longitude'] = 121
            c_c.loc[c,'Latitude'] = 25
        elif c == 'ROW':
            c_c.loc[c,'Longitude'] = 0
            c_c.loc[c,'Latitude'] = 0
        else:
            c_c.loc[c,'Longitude'] = world.set_index('country').loc[c,'longitude']
            c_c.loc[c,'Latitude'] = world.set_index('country').loc[c,'latitude']

    c_c = c_c.reset_index().T.reset_index().T
    c_c.iloc[0] = c_c.iloc[0]+' , '+c_c.iloc[1]
    c_c.drop('Quantity',inplace = True)

    return c_c

# Write csv
cons_network = network_create(cons_flows)
cons_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/cons_network.csv')

iot_network = network_create(iot_flows)
iot_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/iot_network.csv')

total_network = network_create(trade_flows)
total_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/total_network.csv')

#%% Obtain mean value of change for maps
# cons_flows['value'] = cons_flows['value_traded'] / (cons_flows['value_output'] * cons_flows['value_imports'])
# cons_flows['new'] = cons_flows['new_traded'] / (cons_flows['new_output'] * cons_flows['new_imports'])
# cons_flows['change'] = (cons_flows['new'] / cons_flows['value'] -1)*100
# cons_flows['change_output'] = (cons_flows['new_output'] / cons_flows['value_output'] -1)*100

cons_network.T.set_index('Sector').T

cons_network[cons_network[' Total , Trade flow weighted change (%)'].isnull()][' Total , Trade flow weighted change (%)'].mean()