#%% Import libraries
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
# import treatment_funcs as t
import treatment_funcs_agri_ind_fe as t

#%% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

#%% Calculating effective tax rate

# Load solution : Baseline = 2018, carbon cost = 100
print('Setting parameters for run')

y = 2018
year = str(y)
dir_num = 8
path = 'results/'+year+'_'+str(dir_num)
carb_cost = 1e-4

print('Loading '+year+' data, baseline for carbon tax '+str(carb_cost*1e6)+' dollar per ton of CO2')
runs = pd.read_csv(path+'/runs')

# Baseline data
cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b = t.load_baseline(year)
sh = t.shares(cons_b, iot_b, output_b, va_b, co2_prod_b, co2_intensity_b)

sector_list = iot_b.index.get_level_values(1).drop_duplicates().to_list()
S = len(sector_list)
country_list = iot_b.index.get_level_values(0).drop_duplicates().to_list()
C = len(country_list)

cons_traded_unit = cons_b.reset_index()[['row_country','row_sector','col_country']]
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit['row_country'] == cons_traded_unit['col_country'] , ['value','new']] = 0
cons_traded_unit_np = cons_traded_unit.value.to_numpy()
iot_traded_unit = iot_b.reset_index()[['row_country','row_sector','col_country','col_sector']]
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit['row_country'] == iot_traded_unit['col_country'] , ['value','new']] = 0
iot_traded_unit_np = iot_traded_unit.value.to_numpy()

sol_all = {}
if y not in sol_all.keys():
    print('Baseline year ' + str(y))
    sol_all[y] = t.sol(y, dir_num, carb_cost)

run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

#%% Total spending for effective tax rate
# Compute aggregate spending, and spending weighted by carbon intensity
m_df = sol_all[y].iot.groupby(level=[0,1,2]).sum().new
s_df = pd.merge(
    sol_all[y].cons.new,
    m_df,
    right_index=True,
    left_index=True,
    suffixes=('_cons','_m')
)
s_df.reset_index(inplace=True)
s_df = s_df.merge(
    sol_all[y].co2_intensity.reset_index(),
    how='left',
    left_on=['row_country', 'row_sector'],
    right_on=['country', 'sector']
)
s_df.drop(['country', 'sector'], axis=1, inplace=True)

s_df['spending'] = s_df['new_cons'] + s_df['new_m']
s_df['spending_carb'] = s_df['spending']*s_df['value']*carb_cost      # Tons of CO2 * millions of dollars / Tons

# Aggregate at the consuming country level
sagg_df = s_df.groupby(['col_country']).sum()[['spending', 'spending_carb']]
sagg_df['tax'] = sagg_df['spending_carb']/sagg_df['spending']*100

# Summarise
sagg_df['tax'].describe()

# Bar Plot
sagg_df.reset_index(inplace=True)
sagg_df.sort_values('tax', ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(sagg_df.col_country
            ,sagg_df.tax
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Effective tax rate on spending',
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
ax.set_ylabel('% of total spending', fontsize = 20)

ax.bar_label(ax.containers[0],
             labels=sagg_df.col_country,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

ax.set_ylim(0,9.5)
plt.show()

#%%  Effective transfers (just renaming wage GDP-per-capita for clarity)

# Load population data
print('Loading labor data')

labor = pd.read_csv('data/World bank/labor_force/labor.csv')
labor.set_index('country', inplace = True)
labor.sort_index(inplace = True)
labor_year = labor[year]

# Obtain price index from loaded data
cons, iot, output, va, co2_prod, price_index = t.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,
                                                                          co2_prod_b, sh)

print('Plotting adjustments needed for a fair tax')

gdp = sol_all[y].va.groupby(level=0).sum()
gdp['price_index'] = price_index

gdp = gdp.join(labor_year).rename(columns={year:'labor'})
gdp['per_capita'] = (gdp.value / gdp.labor)

gdp['change_real'] = ((gdp.new/gdp.value)/(gdp.price_index))

# gdp['new_adjusted'] = gdp['change_real']*gdp['value']
gdp['new_adjusted'] = gdp['new']

gdp_mean_change = gdp['new_adjusted'].sum()/gdp['value'].sum()

gdp['new_if_average_change_adjusted']  = gdp['value'] * gdp_mean_change

gdp['contribution'] = gdp['new_adjusted'] - gdp['new_if_average_change_adjusted']

gdp['wage_change_for_equality'] = -(gdp['contribution'] / gdp.labor)*1e6

gdp['relative_change'] = (gdp['wage_change_for_equality'] / (gdp.per_capita*1e6))*100

gdp.sort_values('wage_change_for_equality',inplace=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout=True)

ax1=ax.twinx()

ax.bar(gdp.index.get_level_values(0),gdp['wage_change_for_equality'])

ax.set_xticklabels([''])
ax.tick_params(axis = 'y', colors=sns.color_palette()[0], labelsize = 20)
ax.set_ylabel('Contribution per worker ($)', color = sns.color_palette()[0] , fontsize = 30)

ax.bar_label(ax.containers[0],
             labels=gdp.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

ax1.scatter(gdp.index.get_level_values(0),gdp['relative_change'], color=sns.color_palette()[1])

ax1.grid(visible=False)
ax.margins(x=0.01)
ax.set_ylim(-6000,6000)
ax1.set_ylim(-6,6)

ax1.tick_params(axis = 'y', colors=sns.color_palette()[1] , labelsize = 20 )
ax1.set_ylabel('Contribution per worker (% of GDPpc)', color = sns.color_palette()[1] , fontsize = 30)


plt.show()

# Calculate average transfer
pos_total = gdp.loc[gdp['contribution']>0, 'contribution'].sum()
posL_total = gdp.loc[gdp['contribution']>0, 'labor'].sum()
neg_total = gdp.loc[gdp['contribution']<=0, 'contribution'].sum()
negL_total = gdp.loc[gdp['contribution']<=0, 'labor'].sum()

print('Positive transfer per capita:', pos_total/posL_total*1e6)
print('Negative transfer per capita:', neg_total/negL_total*1e6)