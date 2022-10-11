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
# cons_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/cons_network.csv')

iot_network = network_create(iot_flows)
# iot_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/iot_network.csv')

total_network = network_create(trade_flows)
# total_network.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Carbon_tax_model/networks/Test_weights/total_network.csv')

#%% Obtain mean value of change for maps
# cons_flows['value'] = cons_flows['value_traded'] / (cons_flows['value_output'] * cons_flows['value_imports'])
# cons_flows['new'] = cons_flows['new_traded'] / (cons_flows['new_output'] * cons_flows['new_imports'])
# cons_flows['change'] = (cons_flows['new'] / cons_flows['value'] -1)*100
# cons_flows['change_output'] = (cons_flows['new_output'] / cons_flows['value_output'] -1)*100

cons_network.T.set_index('Sector').T
cons_network[cons_network[' Total , Trade flow weighted change (%)'].isnull()][' Total , Trade flow weighted change (%)'].mean()

#%% Trade reallocation

#%% Trade reallocation - Sector-wise - TOTAL EFFECT

print('Computing trade flows reallocation sector-wise')

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

# Construct dataframe
sector_change = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
    temp = traded[y].groupby(level=[0,1,2]).sum().xs(sector,level=1).new-traded[y].groupby(level=[0,1,2]).sum().xs(sector,level=1).value
    # temp = traded[y].groupby(level=[1,2]).sum().xs(sector,level=0).new-traded[y].groupby(level=[1,2]).sum().xs(sector,level=0).value
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

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

print('Overall, '+str(total_output_reallocated_percent.round(2))+'% of traded volumes \nwould be reallocated within a sector \nacross countries for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
          )

#%% Production reallocation, nominal differences

print('Plotting trade reallocation in nominal differences')

sector_org = sector_dist_df[['industry', 'change', 'realloc','realloc_share_nom', 'change_tot_nom']].copy()
sector_pos = sector_org[sector_org['realloc_share_nom']>0].copy()
sector_pos.sort_values('change', ascending = True, inplace = True)
sector_neg1 = sector_org[sector_org['realloc_share_nom']<= -0.15].copy()
sector_neg1.sort_values('change',ascending = True, inplace = True)
sector_neg2 = sector_org[(sector_org['realloc_share_nom']> -0.15) & (sector_org['realloc_share_nom']<=0)].copy()
sector_neg2.sort_values('change',ascending = True, inplace = True)

sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change/1e6
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of trade flows',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc/1e6
            ,bottom = sector_use.change/1e6
            ,label='Reallocated trade flows',
            # color=colors,
            hatch="////")

# ax.set_xticklabels(sector_dist_df.industry
#                     , rotation=45
#                     , ha='right'
#                     , rotation_mode='anchor'
#                     ,fontsize=19)

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Trillion $', fontsize = 20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of traded volumes \nwould be reallocated within a sector \nacross countries for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(27,-0.28),fontsize=25,zorder=10,backgroundcolor='w')

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

#%% trade reallocation, % changes

print('Plotting trade reallocation in percentages')

# sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg', 'change_tot']].copy()
# sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
# sector_pos.sort_values('change_percent', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
# sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
# sector_neg2.sort_values('change_percent',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)
sector_use = sector_dist_df.sort_values('change_percent', ascending=True)


fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change in trade volumes (%)',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc_percent
            ,bottom = sector_use.change_percent
            ,label='Reallocated trade (%)',
            # color=colors,
            hatch="////")

# ax.set_xticklabels(sector_dist_df.industry
#                     , rotation=45
#                     , ha='right'
#                     , rotation_mode='anchor'
#                     ,fontsize=19)

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial trade volumes', fontsize = 20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross trade volumes \nwould be reallocated within a sector \nacross country pairs for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
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

#%% Trade reallocation - Sector-wise - Origin/Destination effect

print('Computing trade flows reallocation sector-wise')

Origin = False

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

# Construct dataframe
sector_change = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
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

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

print('Overall, '+str(total_output_reallocated_percent.round(2))+'% of traded volumes \nwould be reallocated within a sector \nacross countries for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
          )

#%% Production reallocation, nominal differences

print('Plotting trade reallocation in nominal differences')

sector_org = sector_dist_df[['industry', 'change', 'realloc','realloc_share_nom', 'change_tot_nom']].copy()
sector_pos = sector_org[sector_org['realloc_share_nom']>0].copy()
sector_pos.sort_values('change', ascending = True, inplace = True)
sector_neg1 = sector_org[sector_org['realloc_share_nom']<= -0.15].copy()
sector_neg1.sort_values('change',ascending = True, inplace = True)
sector_neg2 = sector_org[(sector_org['realloc_share_nom']> -0.15) & (sector_org['realloc_share_nom']<=0)].copy()
sector_neg2.sort_values('change',ascending = True, inplace = True)

sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change/1e6
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of trade flows',
            # color=colors
            )

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

# ax.set_xticklabels(sector_dist_df.industry
#                     , rotation=45
#                     , ha='right'
#                     , rotation_mode='anchor'
#                     ,fontsize=19)

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Trillion $', fontsize = 20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

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

#%% trade reallocation, % changes

print('Plotting trade reallocation in percentages')

# sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg', 'change_tot']].copy()
# sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
# sector_pos.sort_values('change_percent', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
# sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
# sector_neg2.sort_values('change_percent',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)
sector_use = sector_dist_df.sort_values('change_percent', ascending=True)


fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change in trade volumes (%)',
            # color=colors
            )

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

# ax.set_xticklabels(sector_dist_df.industry
#                     , rotation=45
#                     , ha='right'
#                     , rotation_mode='anchor'
#                     ,fontsize=19)

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial trade volumes', fontsize = 20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# if Origin:
#     ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross trade volumes \nwould be reallocated within a sector \nacross origins for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-25),fontsize=25,zorder=10,backgroundcolor='w')
# else:
#     ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross trade volumes \nwould be reallocated within a sector \nacross destinations for a net reduction \nof trade flows of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-25),fontsize=25,zorder=10,backgroundcolor='w')

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


#%% Country specific reallocation of exports/imports - TOTAL EFFECT

print('Computing trade reallocation country-wise - Origin/Exporting country or Destination/Importing country - TOTAL EFFECT')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_change = []
country_realloc_pos = []
country_realloc_neg = []

Origin = False
# Sector = False

if Origin:
    for country in country_list:
        # print(country)
        temp = traded[y].groupby(level=[0,1,2]).sum().xs(country,level=0).new-traded[y].groupby(level=[0,1,2]).sum().xs(country,level=0).value
        country_change.append(temp.sum())
        country_realloc_pos.append(temp[temp>0].sum())
        country_realloc_neg.append(temp[temp<0].sum())
        country_dist_df = pd.DataFrame(index=country_list)
        country_dist_df['traded'] = traded[y].groupby(level=0).sum().value.values
        country_dist_df['traded_new'] = traded[y].groupby(level=0).sum().new.values
else:
    for country in country_list:
        temp = traded[y].groupby(level=[0,1,2]).sum().xs(country, level=2).new - traded[y].groupby(level=[0,1,2]).sum().xs(country, level=2).value
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

print('Plotting production reallocation in percentages - TOTAL EFFECT')

country_dist_df.sort_values('change_percent',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()
if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.change_percent
                # ,bottom = country_dist_df.realloc_neg
                ,label='Net change in exports (%)',
                # color=colors
                )

    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.realloc_percent
                ,bottom = country_dist_df.change_percent
                ,label='Reallocated exports across sectors and destinations (%)',
                # color=colors,
                hatch="////")

else:
    ax.bar(country_dist_df.index.get_level_values(0)
           , country_dist_df.change_percent
           # ,bottom = country_dist_df.realloc_neg
           , label='Net change in imports (%)',
           # color=colors
           )

    ax.bar(country_dist_df.index.get_level_values(0)
           , country_dist_df.realloc_percent
           , bottom=country_dist_df.change_percent
           , label='Reallocated imports across sectors and origins (%)',
           # color=colors,
           hatch="////")

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

if Origin:
    ax.set_ylabel('% of initial exports',
              fontsize = 20)
else:
    ax.set_ylabel('% of initial imports',
                  fontsize=20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=country_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(country_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)



leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-20,5)

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors or destinations\nfor a net reduction of (real) output \nof '+str(total_output_decrease_percent.round(2))+'%',
             xy=(41,-25),fontsize=25,zorder=10,backgroundcolor='w')


max_lim = country_dist_df['total_change'].max()
min_lim = country_dist_df['total_change'].min()
ax.set_ylim(min_lim-2,max_lim+2)

plt.show()



#%% Country specific reallocation of exports/imports

print('Computing trade reallocation country-wise - Origin/Exporting country or Destination/Importing country')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_change = []
country_realloc_pos = []
country_realloc_neg = []

Origin = False
Sector = False

if Origin:
    for country in country_list:
        # print(country)
        if Sector:
            temp = traded[y].groupby(level=[0,1]).sum().xs(country,level=0).new-traded[y].groupby(level=[0,1]).sum().xs(country,level=0).value
            country_change.append(temp.sum())
            country_realloc_pos.append(temp[temp>0].sum())
            country_realloc_neg.append(temp[temp<0].sum())
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
        if Sector:
            temp = traded[y].groupby(level=[1,2]).sum().xs(country, level=1).new - traded[y].groupby(level=[1,2]).sum().xs(country, level=1).value
            country_change.append(temp.sum())
            country_realloc_pos.append(temp[temp > 0].sum())
            country_realloc_neg.append(temp[temp < 0].sum())
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


#%% Reallocation of production, nominal differences

print('Plotting production reallocation in nominal differences')

country_dist_df.sort_values('change_tot_nom',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()
if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.change/1e6
                # ,bottom = country_dist_df.realloc_neg
                ,label='Net change in exports',
                # color=colors
                )
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
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Trillion $',
              fontsize = 20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=country_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(country_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)



leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-1.85,0.25)

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

if Origin:
    if Sector:
        ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of trade volumes \nwould be reallocated within an \nexporting country across sectors \nfor a net reduction of trade flows \nof '+str(total_output_decrease_percent.round(2))+'%',
             xy=(41,-0.2),fontsize=25,zorder=10,backgroundcolor='w')
    else:
        ax.annotate('Overall, ' + str(total_output_reallocated_percent.round(
            2)) + '% of trade volumes \nwould be reallocated within an \nexporting country across destinations \nfor a net reduction of trade flows \nof ' + str(
            total_output_decrease_percent.round(2)) + '%',
                    xy=(41, -0.12), fontsize=25, zorder=10, backgroundcolor='w')
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

#%% Reallocation of production, percentage changes

print('Plotting production reallocation in percentages')

country_dist_df.sort_values('change_percent',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()
if Origin:
    ax.bar(country_dist_df.index.get_level_values(0)
                ,country_dist_df.change_percent
                # ,bottom = country_dist_df.realloc_neg
                ,label='Net change in exports (%)',
                # color=colors
                )
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
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

if Origin:
    ax.set_ylabel('% of initial exports',
              fontsize = 20)
else:
    ax.set_ylabel('% of initial imports',
                  fontsize=20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=country_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(country_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)



leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-20,5)

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

total_output_net_decrease = traded[y].value.sum() - traded[y].new.sum()
total_output = traded[y].value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(41,-20),fontsize=25,zorder=10,backgroundcolor='w')


max_lim = country_dist_df['total_change'].max()
min_lim = country_dist_df['total_change'].min()
ax.set_ylim(min_lim-2,max_lim+2)

plt.show()

#%% Test to approximate changes in terms of trade
# Import prices are aggregate consumer price index = price_index
# Export prices are price deflator on total exports - approximate its change as a
# weighted average of sectoral price changes (factory based since tau doesn't move)
# where weights are export share in the baseline (i.e. ignoring quantity adjustments)

price_hat = sol_all[y].res.price_hat
# Obtain price index from loaded data
cons, iot, output, va, co2_prod, price_index, utility = t.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,
                                                                      co2_prod_b, sh)

cons_adj = pd.merge(
    sol_all[y].cons.value.reset_index(),
    price_hat.reset_index(),
    'left',
    left_on=['row_country', 'row_sector'],
    right_on=['country', 'sector'],
)[
    ['row_country', 'row_sector', 'col_country', 'value', 'price_hat']
].set_index(['row_country', 'row_sector', 'col_country'])
cons_adj['cons'] = cons_adj['value']*cons_adj['price_hat']
cons_adj = cons_adj[['cons']].groupby(level=[0,1]).sum()

iot_adj = pd.merge(
    sol_all[y].iot.value.reset_index(),
    price_hat.reset_index(),
    'left',
    left_on=['row_country', 'row_sector'],
    right_on=['country', 'sector'],
)[
    ['row_country', 'row_sector', 'col_country', 'col_sector', 'value', 'price_hat']
].set_index(['row_country', 'row_sector', 'col_country', 'col_sector'])
iot_adj['iot'] = iot_adj['value']*iot_adj['price_hat']
iot_adj = iot_adj[['iot']].groupby(level=[0,1]).sum()

output_adj = pd.DataFrame(
    cons_adj.cons + iot_adj.iot, columns=['new_adj']
).rename_axis(['country', 'sector']).join(sol_all[y].output)

temp = output_adj.groupby(level=0).sum()
ToT_df = pd.DataFrame(
    temp.new_adj / temp.value, columns=['deflator']
)

ToT_df['price_index'] = price_index
ToT_df['ToT_hat'] = ToT_df['deflator'] / ToT_df['price_index']
ToT_df['ToT_hat_percent'] = (ToT_df['ToT_hat'] - 1)*100

#%% Bar plot ToT change
ToT_use = ToT_df.reset_index()
ToT_use.sort_values('ToT_hat_percent', ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

ax.bar(ToT_use.country
       , ToT_use.ToT_hat_percent
       # ,bottom = sector_dist_df.realloc_neg
       , label='Change (%) in Terms of Trade',
       # color=colors
       )

ax.set_xticklabels(['']
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   , fontsize=19)

ax.tick_params(axis='x', which='major', labelsize=20, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('% change', fontsize=20)

ax.bar_label(ax.containers[0],
             labels=ToT_use.country,
             rotation=90,
             label_type='edge',
             padding=2,
             zorder=10)
max_lim = ToT_use.ToT_hat_percent.max()
min_lim = ToT_use.ToT_hat_percent.min()
ax.set_ylim(min_lim-0.9, max_lim+1)
plt.show()

#%% Connectivity measure
print('Computing connectivities')

real = False
if real:
    c_c = tot[y].join(price_hat)
    c_c['new'] = c_c.new / c_c.price_hat
    c_c.drop('price_hat', axis=1, inplace=True)
    c_c = c_c.groupby(level=[0,2]).sum()
else:
    c_c = tot[y].groupby(level=[0,2]).sum()

c_co2_intensity = sol_all[y].co2_prod.groupby(level=0).sum().value / sol_all[y].output.groupby(level=0).sum().value
exports = c_c.groupby(level=0).sum().rename_axis('country')
imports = c_c.groupby(level=1).sum().rename_axis('country')

# Compute self share as share of total exports consumed locally
own_trade = c_c.reset_index()[c_c.reset_index().row_country == c_c.reset_index().col_country]
own_trade = own_trade.drop('col_country',axis=1).rename(columns={'row_country' : 'country'})
own_trade = own_trade.merge(imports.reset_index(),suffixes = ['','_total_exchange'],on='country')

own_trade['value_self'] = own_trade.value / own_trade.value_total_exchange
own_trade['new_self'] = own_trade.new / own_trade.new_total_exchange
own_trade['value_connect'] = 1 - own_trade.value_self
own_trade['new_connect'] = 1 - own_trade.new_self
own_trade['change_connect'] = (own_trade['new_connect'] / own_trade['value_connect'] -1)*100
own_trade.set_index('country' , inplace=True)
own_trade['co2_intensity'] = c_co2_intensity.values*1e6

country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')
own_trade = own_trade.join(labor_year).rename(columns={year:'labor'})

own_trade = own_trade.join(country_map)
own_trade.loc['TWN','Continent'] = 'Asia'
own_trade.loc['ROW','Continent'] = 'Africa'
own_trade.loc['AUS','Continent'] = 'Asia'
own_trade.loc['NZL','Continent'] = 'Asia'
own_trade.loc['CRI','Continent'] = 'South America'
own_trade.loc['RUS','Continent'] = 'Asia'
own_trade.loc['SAU','Continent'] = 'Africa'
own_trade.loc['GRC','labor'] = own_trade.loc['GRC','labor']*0.5

#%% Connectivity as a function of CO2 intensity
variable = 'change_connect'

print('Plotting connectivity as a function of CO2 intensity')

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }
colors = [continent_colors[own_trade.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

ax.scatter(own_trade['co2_intensity'] , own_trade[variable],lw=3,s=50,marker='x',c=colors)

sns.kdeplot(data=own_trade,
                x='co2_intensity',
                y=variable,
                hue = 'Continent',
                fill = True,
                alpha = 0.2,
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                # weights = 'labor',
                # legend=False,
                levels = 2,
                palette = palette,
                # common_norm = False,
                shade=True,
                thresh = 0.1,
                # dropna=True,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )

sns.move_legend(ax,'lower right')

coeffs_fit = np.polyfit(own_trade['co2_intensity'],
                  own_trade[variable],
                  deg = 1,
                  # w=own_trade.labor
                  )

x_lims = (0,900)
# y_lims = (-0.0025*200,0.0025*200)
ax.set_xlim(*x_lims)
y_lims = (-6,6)
ax.set_ylim(*y_lims)


x_vals = np.arange(0,x_lims[1])
y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')

ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')

ax.set_xlabel('CO2 intensity of production (Ton CO2 / $Mio.)',fontsize = 20)
ax.set_ylabel('Change in export share (%)',fontsize = 20)

# plt.legend(loc='lower right')
# ax.legend(loc='lower right')

texts = [plt.text(own_trade['co2_intensity'].loc[country],  own_trade[variable].loc[country], country,size=15,color=colors[i]) for i,country in enumerate(country_list)]
adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

plt.show()

#%% Trade shares changes (share of output traded)

# By sector
print('Computing share of output traded sector-wise')
trade_sh = traded[y].groupby(level=[1]).sum()
trade_sh = trade_sh.div(tot[y].groupby(level=[1]).sum())*100


# Construct dataframe
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

#%% Plot pp difference

print('Plotting sectoral share of output traded changes by sector in pp')

sector_use = sector_dist_df.sort_values('diff', ascending=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

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

# leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['diff'].max()
min_lim = sector_dist_df['diff'].min()
ax.set_ylim(min_lim-0.1,max_lim+0.15)

plt.show()

#%% trade reallocation, % changes

print('Plotting change in sectoral share of output traded in percentages')

sector_use = sector_dist_df.sort_values('diff_pc', ascending=True)


fig, ax = plt.subplots(figsize=(16,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

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

# plt.savefig('../tax_model/eps_figures_for_ralph_pres/cross_sector_effects_trade.eps', format='eps')

plt.show()

#%% Correlation between change in share of traded output and sector's relative dirtyness
print('Computing share of output traded sector-wise')
trade_sh = traded[y].groupby(level=[1]).sum()
trade_sh = trade_sh.div(tot[y].groupby(level=[1]).sum())*100

# Construct dataframe
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

# Calculate production intensity at the sector level
s_co2_intensity = sol_all[y].co2_prod.groupby(level=1).sum().value / sol_all[y].output.groupby(level=1).sum().value
sector_dist_df['co2_int'] = s_co2_intensity.values*1e6

#%%

data = sector_dist_df.copy()
# If excluding outliers
# data = data[data['co2_int']<1000]

sector_list = data.sort_values('group_code').industry.to_list()
data = data.reset_index().set_index('industry').sort_values('group_code')

palette = [sns.color_palette('bright')[i] for i in [2,4,0,3,1,7]]
palette[0] = sns.color_palette()[2]
palette[1] = sns.color_palette("hls", 8)[-2]

sector_colors = {
    'Agro-food' : palette[0],
    'Raw materials' : palette[1],
    'Manufactures' : palette[2],
    'Energy' : palette[3],
    'Services ' : palette[4],
    'Logistics' : palette[5],
                    }
colors = [sector_colors[data.sort_values('group_code').loc[industry,'group_label']] for industry in sector_list]

# data_no_z =data.copy()
# data_no_z = data_no_z[data_no_z['co2_int'] != 0]
# # data_no_z = data_no_z[data_no_z['co2_intensity'] < 1e4]
# # data_no_z['co2_intensity'] = np.log(data_no_z['co2_intensity'])
# data_no_z = data_no_z[['co2_intensity','value','group_label','group_code','output']]
#
# data_no_z_1 = data_no_z[data_no_z['co2_int'] < 100].copy()
# data_no_z_2 = data_no_z[data_no_z['co2_int'] >= 100].copy()

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)
# for i,group in enumerate(data_no_z_i.group_code.drop_duplicates().to_list()):
for i,group in enumerate(data.group_code.drop_duplicates().to_list()):
    ax.scatter(data[data['group_code'] == group].co2_int,data[data['group_code'] == group].diff_pc,s=50, color=palette[i], marker='x',lw=2,zorder=1-i)

# ax.scatter(sector_dist_df.co2_int,sector_dist_df.diff_pc,marker='x',lw=2,s=50)

# sns.kdeplot(data=sector_dist_df,
#                 x='co2_int',
#                 y="diff_pc",
#                 hue = 'group_code',
#                 fill = True,
#                 alpha = 0.2,
#                 # height=10,
#                 # ratio=5,
#                 # bw_adjust=0.7,
#                 # weights = 'labor',
#                 # legend=False,
#                 levels = 2,
#                 palette = palette,
#                 # log_scale = (True, False)
#                 # common_norm = False,
#                 shade=True,
#                 thresh = 0.15,
#                 # dropna=True,
#                 # fill = False,
#                 # alpha=0.6,
#                 # hue_order = sector_dist_df.group_label.drop_duplicates().to_list()[::-1],
#                 ax = ax
#                 )

coeffs_fit = np.polyfit(data.co2_int,
                  data.diff_pc,
                  deg = 1,
                  #w=emissions.labor
                  )

ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)', fontsize = 20)
ax.set_ylabel('Change in share of traded output (%)', fontsize = 20)

# ax.set_yscale('log')
ax.set_xscale('log')
# x_lims = (10,2000)
x_lims = (10,4000)
ax.set_xlim(*x_lims)
# y_lims = (-5,10)
y_lims = (-15,35)
ax.set_ylim(*y_lims)

x_vals = np.arange(0,x_lims[1])
y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')

ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')

# texts = [plt.text(sector_dist_df['co2_int'].loc[industry],  sector_dist_df['diff_pc'].loc[industry], industry,size=15,color=colors[i]) for i,industry in enumerate(sector_list)]
# adjust_text(texts,
#             precision=0.001,
#             expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#             force_text=(0.01, 0.25), force_points=(0.01, 0.25)
#             , arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

sect = 'Energy'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(50, 15),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Mining, energy'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(80, 15),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Basic metals'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(80, 15),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Agriculture'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(50, 15),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Electronic'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(80, 35),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Air transport'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(80, -15),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Machinery'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(-80, -55),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sect = 'Food products'
sect_index = sector_list.index(sect)
ax.annotate(sect,
            xy=(data.loc[sect].co2_int, data.loc[sect].diff_pc),
            xycoords='data',
            xytext=(-80, -55),
            textcoords='offset points',
            va='center',
            color=colors[sect_index],
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

plt.show()

#%% Country specific reallocation of exports/imports

print('Computing trade reallocation country-wise - Origin/Exporting country or Destination/Importing country')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

Origin = False

if Origin:
    trade_sh = traded[y].groupby(level=[0]).sum()
    trade_sh = trade_sh.div(tot[y].groupby(level=[0]).sum()) * 100
else:
    trade_sh = traded[y].groupby(level=[2]).sum()
    trade_sh = trade_sh.div(tot[y].groupby(level=[2]).sum()) * 100

country_dist_df = pd.DataFrame(index=country_list)
country_dist_df['value'] = trade_sh.value.values
country_dist_df['new'] = trade_sh.new.values
country_dist_df['diff'] = country_dist_df.new - country_dist_df.value
country_dist_df['diff_pc'] = country_dist_df['diff'] /country_dist_df['value']*100


#%%
print('Plotting change in export/import share in pp')

country_dist_df.sort_values('diff',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df['diff']
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
ax.set_ylabel('Percentage points',
              fontsize = 20)

# leg = ax.legend(fontsize=20,loc='lower right')

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

#%%Plotting change in import/export share in percentages
print('Plotting change in import/export share in percentages')

country_dist_df.sort_values('diff_pc',inplace = True)

fig, ax = plt.subplots(figsize=(16,10),constrained_layout = True)

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

# leg = ax.legend(fontsize=20,loc='lower right')

ax.grid(axis='x')
# ax.set_ylim(-20,5)

ax.bar_label(ax.containers[0],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

max_lim = country_dist_df['diff_pc'].max()
min_lim = country_dist_df['diff_pc'].min()
ax.set_ylim(min_lim-1,max_lim+1)

# plt.savefig('../tax_model/eps_figures_for_ralph_pres/cross_country_effects_trade.eps', format='eps')

plt.show()


#%% Checks - Production composition
country = 'SAU'
variable = 'change_share'
print('Plotting gross output composition for '+country)

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


fig, ax = plt.subplots(figsize=(18,10))
color = sns.color_palette()[7]

data = sc_df.xs(country, level=0)[variable].to_list()
indicators = sector_map.group_code.to_list()
group_labels = sector_map.group_label.to_list()

indicators_sorted , sector_list_full_sorted , data_sorted , group_labels_sorted  = zip(*sorted(zip(indicators, sector_list_full, data , group_labels)))

group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# palette[6] , palette[7] = palette[7] , palette[6]
colors = [palette[ind-1] for ind in indicators_sorted]

#ordered with groups
ax.bar(sector_list_full_sorted, data_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
ax.set_xticklabels(sector_list_full_sorted
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.tick_params(axis='x', which='major', pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)

# handles = []
# for ind in indicators_sorted:
handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax.legend(handles=handles,fontsize=20)


# ax.legend(group_labels_sorted)

plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
plt.suptitle('Sectoral output composition in country '+country+' (%)',size = 30,y=0.96)

plt.tight_layout()

plt.show()

# %% Check - Main producer of a sector
sector = '01T02'
variable = 'new'

check_ps = sol_all[y].output.copy()
check_ps = check_ps.xs(sector,level=1).sort_values(variable, ascending=False)

#%% Check main exporter / importer of a sector
sector = '35'
variable = 'new'
Exports = False

if Exports:
    check_ts = traded[y].groupby(level=[0,1]).sum().copy()
    check_ts = check_ts.xs(sector, level=1).sort_values(variable, ascending=False)
else:
    check_ts = traded[y].groupby(level=[1,2]).sum().copy()
    check_ts = check_ts.xs(sector, level=0).sort_values(variable, ascending=False)


# %% Check sectoral composition of exports / imports
country = 'BRA'
Exports = True
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting sectoral export composition for '+country)
else:
    print('Plotting sectoral import composition for ' + country)

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

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)


fig, ax = plt.subplots(2,1,figsize=(18,14))
color = sns.color_palette()[7]

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
# palette[6] , palette[7] = palette[7] , palette[6]
colors = [palette[ind-1] for ind in indicators_sorted]

#ordered with groups
ax[0].bar(sector_list_full_sorted, data1_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
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

# handles = []
# for ind in indicators_sorted:
handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax[0].legend(handles=handles,fontsize=20)

ax[1].bar(sector_list_full_sorted, data2_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
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

# ax[1].legend(handles=handles,fontsize=20)

# ax.legend(group_labels_sorted)

# plt.title('(Tax = $100/Ton of CO2)',size = 25,color=color)
# if Exports:
#     plt.suptitle('Sectoral exports composition in country '+country+' (% and % change)',size = 30,y=0.96)
# else:
#     plt.suptitle('Sectoral imports composition in country '+country+' (% and % change)',size = 30,y=0.96)

plt.tight_layout()

plt.show()

#%% Origin / Destination composition of imports / exports
#(ie where do you export to, from where do you import) 

country = 'IRL'
Exports = False
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting geographic composition of export destinations for '+country)
else:
    print('Plotting geographic composition of import origins for ' + country)

# Construct orig/dest shares
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

#ordered with groups
fig, ax = plt.subplots(2,1, figsize=(18,14))

ax[0].bar(country_t_list_sorted, data1_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
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

#%% Origin / Destination composition of trade volumes for a given sector
#(ie where do you export to, from where do you import)

sector = '24'
Exports = False
variable1 = 'value_share'
variable2 = 'change_share'

if Exports:
    print('Plotting geographic origin of trade for sector '+sector)
else:
    print('Plotting geographic destination of trade for sector' +sector)

# Construct orig/dest shares
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

#ordered with groups
fig, ax = plt.subplots(2,1, figsize=(18,14))

ax[0].bar(country_t_list_sorted, data1_sorted, color=colors,width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
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