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

# %% Aggregate price index change at 100$ tax (weighted average)

# Obtain price index from loaded data
cons, iot, output, va, co2_prod, price_index, utility = t.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,
                                                                      co2_prod_b, sh)

# Calculate aggregation
price_agg_change = (price_index * sh['cons_tot_np'] / sh['cons_tot_np'].sum()).sum()
print('Aggregate price index change - Weighted average:',
      price_agg_change)

# %% Relative sectoral prices in a given country : adjust by local wage - FIG 5
country = 'CHE'
print('Plotting sectoral price indices for country ' + country)

# Construct sectoral price index
p_hat_sol = sol_all[y].res.price_hat.to_numpy().reshape(C, S)

taxed_price = p_hat_sol * (1 + carb_cost * sh['co2_intensity_np'])
price_agg_no_pow = np.einsum('it,itj->tj'
                             , taxed_price ** (1 - sigma)
                             , sh['share_cons_o_np']
                             )
price_agg = np.divide(1,
                      price_agg_no_pow,
                      out=np.ones_like(price_agg_no_pow),
                      where=price_agg_no_pow != 0) ** (1 / (sigma - 1))

# Construct local wage and wage change
wage = sol_all[y].va.groupby(level=0).sum().reset_index()
wage = wage.merge(
    labor_year.reset_index(),
    'left',
    left_on='col_country',
    right_on='country'
)
wage['value'] = wage['value'] / wage['2018'] * 1e6
wage['new'] = wage['new'] / wage['2018'] * 1e6
wage['hat'] = wage['new'] / wage['value']

wage.drop(['country', '2018'], axis=1, inplace=True)

# Verify wage ranking for good measure
wage.sort_values('value')
wage.sort_values('new')

# Create price change adjusted
wage_adj = np.divide(1,wage.hat.to_numpy())
price_agg_adj = np.einsum('si,i -> si', price_agg, wage_adj)

# Add consumer price index and adjust by change in local wage
wage['price_index'] = price_index
wage['price_index_adj'] = wage['price_index'] / wage['hat']

# Set up industry groups
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

# Plot
fig, ax = plt.subplots(figsize=(18, 10))
color = sns.color_palette()[7]

data = ((price_agg_adj[:, country_list.index(country)] - 1) * 100).tolist()
indicators = sector_map.group_code.to_list()
group_labels = sector_map.group_label.to_list()

indicators_sorted, sector_list_full_sorted, data_sorted, group_labels_sorted = zip(
    *sorted(zip(indicators, sector_list_full, data, group_labels)))

group_labels_sorted = list(dict.fromkeys(group_labels_sorted))

palette = [sns.color_palette()[i] for i in [2, 4, 0, 3, 1, 7]]
# palette[6] , palette[7] = palette[7] , palette[6]
colors = [palette[ind - 1] for ind in indicators_sorted]

# ordered with groups
ax.bar(sector_list_full_sorted, data_sorted, color=colors, width=0.5)
# plt.xticks(rotation=35,fontsize=15)
# ax.set_xticklabels(sector_list_full, rotation=60, ha='right',fontsize=15)
ax.set_xticklabels(sector_list_full_sorted
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   , fontsize=19)
ax.tick_params(axis='x', which='major', pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)

# handles = []
# for ind in indicators_sorted:
handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind, group in
           enumerate(group_labels_sorted)]
ax.legend(handles=handles, fontsize=20)

# ax.legend(group_labels_sorted)

plt.title('(% change relative to wage change)', size=25, color=color)
# plt.suptitle('Sectoral consumer price change in Switzerland (%)', size=30, y=0.96)

plt.tight_layout()

plt.show()
# SAve for KIS
# plt.savefig('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/prices_CHE.eps',format='eps')
# keep = sector_map.copy()
# keep['price_change'] = ((price_agg_adj[:, country_list.index(country)] - 1) * 100).tolist()
# keep.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/prices_CHE.csv')

# %% Scatter plot of gross ouput change with kernel density by coarse industry - FIG 6

print(
    'Plotting scatter plot of output changes for every country x sector according to production intensity with kernel density estimates for categories of sectors')

# p_hat_sol = sol_all[y].res.price_hat.to_numpy().reshape(C,S)
E_hat_sol = sol_all[y].res.output_hat.to_numpy().reshape(C, S)
q_hat_sol = E_hat_sol / p_hat_sol
q_hat_sol_percent = (q_hat_sol - 1) * 100

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data = pd.DataFrame(data=q_hat_sol_percent.ravel(),
                    index=pd.MultiIndex.from_product(
                        [country_list, sector_map.index.get_level_values(level=0).to_list()],
                        names=['country', 'sector']),
                    columns=['value'])
data = data.reset_index().merge(sector_map.reset_index(), how='left', left_on='sector', right_on='ind_code').set_index(
    ['country', 'sector']).drop('ind_code', axis=1)
data['co2_intensity'] = sh['co2_intensity_np'].ravel()
data['output'] = sh['co2_intensity_np'].ravel()
data = data.sort_values('group_code')

sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

group_labels_sorted = data.group_label.drop_duplicates().to_list()

data_no_z = data.copy()
data_no_z = data_no_z[data_no_z['co2_intensity'] != 0]
# data_no_z = data_no_z[data_no_z['co2_intensity'] < 1e4]
# data_no_z['co2_intensity'] = np.log(data_no_z['co2_intensity'])
data_no_z = data_no_z[['co2_intensity', 'value', 'group_label', 'group_code', 'output']]

data_no_z_1 = data_no_z[data_no_z['co2_intensity'] < 100].copy()
data_no_z_2 = data_no_z[data_no_z['co2_intensity'] >= 100].copy()

# %%
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
# # sns.move_legend(plot, "lower left", bbox_to_anchor=(.55, .45), title='Species')

# plot.fig.get_axes()[0].legend(loc='lower left')
# # plt.legend(loc='lower left')

palette = [sns.color_palette('bright')[i] for i in [2, 4, 0, 3, 1, 7]]
palette[0] = sns.color_palette()[2]
palette[1] = sns.color_palette("hls", 8)[-2]
for data_no_z_i in [data_no_z_1, data_no_z_2]:
    plot2 = sns.kdeplot(data=data_no_z_i,
                        x='co2_intensity',
                        y="value",
                        hue='group_label',
                        fill=True,
                        alpha=0.25,
                        log_scale=(True, False),
                        # height=10,
                        # ratio=5,
                        # bw_adjust=0.5,
                        weights = 'output',
                        legend=False,
                        levels=2,
                        palette=palette,
                        common_norm=False,
                        # shade=True,
                        thresh=0.2,
                        # fill = False,
                        # alpha=0.6,
                        # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                        ax=ax
                        )
for data_no_z_i in [data_no_z_1, data_no_z_2]:
    for i, group in enumerate(data_no_z_i.group_code.drop_duplicates().to_list()):
        ax.scatter(data_no_z_i[data_no_z_i['group_code'] == group].co2_intensity,
                   data_no_z_i[data_no_z_i['group_code'] == group].value, color=palette[i], s=10, zorder=1 - i)

ax.set_ylabel('Production changes (%)',
              fontsize=30
              )
ax.set_xscale('log')
# ax.set_ylim(-100,+37.5)
ax.set_ylim(-80, +37.5)
# ax.set_xlim(0.5,20000)
ax.set_xlim(data_no_z.co2_intensity.min(), 3e4)
ax.margins(x=0)
ax.tick_params(axis='both', which='major', labelsize=20)

ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)',
              fontsize=30)

handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind, group in
           enumerate(group_labels_sorted)]
ax.legend(handles=handles, fontsize=20, loc='lower left')

ax.xaxis.set_major_formatter(ScalarFormatter())

ax.hlines(0, xmin=sh['co2_intensity_np'].min(), xmax=1e5, colors='black', ls='--', lw=1)

# sec = '20'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'RUS'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-200, -5),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )
#
# sec = '28'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'CHN'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-150, -100),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )
#
# sec = '35'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'NOR'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(120, 60),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )
#
# sec = '50'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'DEU'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(80, 15),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )
#
# sec = '01T02'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'BRA'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-250, -5),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '01T02'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)
#
# country = 'CHE'
# country_index = country_list.index(country)
#
# ax.annotate(country + ' - ' + sector,
#             xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(100, -15),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

plt.show()
# SAve for KIS
# plt.savefig('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/output_scatter.eps',format='eps')
# keep = data_no_z.copy().drop('output', axis=1).rename(columns={'value':'change'})
# add = sector_map[['industry']].copy()
# add.index.names = ['sector']
# keep = keep.join(add, how='left')
# keep.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/output_scatter.csv')

# %% Labor transfers

# Calculate labor force by country x sector in counterfactual and realized world
labor_sect = pd.merge(
    sol_all[y].va.reset_index(),
    wage,
    'left',
    on='col_country',
    suffixes=('_va', '_wage')
)
labor_sect.drop('hat', axis=1, inplace=True)
labor_sect['value_L'] = labor_sect['value_va'] / labor_sect['value_wage'] * 1e6
labor_sect['new_L'] = labor_sect['new_va'] / labor_sect['new_wage'] * 1e6

# Test they sum up to something equal in both instances
(labor_sect.groupby('col_country').sum().value_L - labor_sect.groupby('col_country').sum().new_L).max()

# Calculate labor gain or loss by sector for each country
labor_sect['diff_L'] = labor_sect['new_L'] - labor_sect['value_L']

# Re-index dataframe
labor_sect.set_index(['col_country', 'col_sector'], inplace=True)

# %% Reallocation of labor by sectors
sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

sector_change = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
    temp = labor_sect.xs(sector, level=1).new_L - labor_sect.xs(sector, level=1).value_L
    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp > 0].sum())
    sector_realloc_neg.append(temp[temp < 0].sum())

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_df = sector_map.copy()
sector_df['value_L'] = labor_sect.groupby(level=1).sum().value_L.values
sector_df['new_L'] = labor_sect.groupby(level=1).sum().new_L.values
sector_df['realloc_pos'] = sector_realloc_pos
sector_df['realloc_neg'] = sector_realloc_neg
sector_df['change'] = sector_change

sector_df['realloc_pos'] = np.abs(sector_df['realloc_pos'])
sector_df['realloc_neg'] = np.abs(sector_df['realloc_neg'])
sector_df['realloc'] = sector_df[['realloc_neg', 'realloc_pos']].min(axis=1)
sector_df['realloc'] = sector_df['realloc'] * np.sign(sector_df['change'])
sector_df['change_tot_nom'] = (sector_df['change'] + sector_df['realloc'])
sector_df['realloc_share_nom'] = (sector_df['realloc'] / sector_df['change_tot_nom']) * np.sign(sector_df['change'])

sector_df['realloc_percent'] = (sector_df['realloc'] / sector_df['value_L']) * 100
sector_df['change_percent'] = (sector_df['change'] / sector_df['value_L']) * 100
sector_df['change_tot'] = (sector_df['change_percent'] + sector_df['realloc_percent'])
sector_df['realloc_share_neg'] = (sector_df['realloc_percent'] / sector_df['change_tot']) * np.sign(sector_df['change'])

# %% Labor reallocation within sectors, number of workers

print('Plotting labor reallocation across countries within sectors in number of workers')

sector_org = sector_df[['industry', 'change', 'realloc', 'realloc_share_nom', 'change_tot_nom']].copy()
# sector_pos = sector_org[sector_org['realloc_share_nom']>0].copy()
# sector_pos.sort_values('change', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_nom']<= -0.15].copy()
# sector_neg1.sort_values('change',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_nom']> -0.15) & (sector_org['realloc_share_nom']<=0)].copy()
# sector_neg2.sort_values('change',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

sector_use = sector_org.sort_values('change', ascending=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
       , sector_use.change / 1e6
       # ,bottom = sector_dist_df.realloc_neg
       , label='Net change of workforce',
       # color=colors
       )

ax.bar(sector_use.industry
       , sector_use.realloc / 1e6
       , bottom=sector_use.change / 1e6
       , label='Reallocated labor',
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
                   , fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize=20, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('Million workers', fontsize=20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20, loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

# total_output_net_decrease = output.value.sum() - output.new.sum()
# total_output = output.value.sum()
# total_output_decrease_percent = (total_output_net_decrease/total_output)*100
#
# total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
# total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated sector-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-1),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
             label_type='edge',
             padding=2,
             zorder=10)

min_lim = sector_use['change_tot_nom'].min() / 1e6
max_lim = sector_use['change_tot_nom'].max() / 1e6
ax.set_ylim(min_lim - 20, max_lim + 20)

plt.show()

# %% Labor reallocation within sectors, % changes

print('Plotting labor reallocation across countries within sectors in % of initial sectoral worforce')

sector_org = sector_df[['industry', 'change_percent', 'realloc_percent', 'realloc_share_neg', 'change_tot']].copy()
# sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
# sector_pos.sort_values('change_percent', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
# sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
# sector_neg2.sort_values('change_percent',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)
sector_use = sector_org.sort_values('change_percent', ascending=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
       , sector_use.change_percent
       # ,bottom = sector_dist_df.realloc_neg
       , label='Net change of workforce (%)',
       # color=colors
       )

ax.bar(sector_use.industry
       , sector_use.realloc_percent
       , bottom=sector_use.change_percent
       , label='Reallocated labor (%)',
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
                   , fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize=20, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial workforce', fontsize=20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=sector_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(sector_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)

leg = ax.legend(fontsize=20, loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

# total_output_net_decrease = output.value.sum() - output.new.sum()
# total_output = output.value.sum()
# total_output_decrease_percent = (total_output_net_decrease/total_output)*100
#
# total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
# total_output_reallocated_percent = (total_output_reallocated/total_output)*100
#
# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated sector-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-22.5),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
             label_type='edge',
             padding=2,
             zorder=10)

min_lim = sector_use['change_tot'].min()
max_lim = sector_use['change_tot'].max()
ax.set_ylim(min_lim - 20, max_lim + 20)

plt.show()

# %% Reallocation of labor - country level disruption
print('Computing production reallocation country-wise')

country_map = pd.read_csv('data/countries_after_agg.csv', sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_realloc_pos = []
country_realloc_neg = []
for country in country_list:
    temp = labor_sect.xs(country, level=0).new_L - labor_sect.xs(country, level=0).value_L
    country_realloc_pos.append(temp[temp > 0].sum())
    country_realloc_neg.append(temp[temp < 0].sum())

country_df = pd.DataFrame(index=country_list)
country_df['labor'] = labor_sect.groupby(level=0).sum().value_L.values
country_df['realloc_pos'] = country_realloc_pos
country_df['realloc_neg'] = country_realloc_neg
country_df['change'] = country_df['realloc_pos'] + country_df['realloc_neg']

country_df['realloc_pos'] = np.abs(country_df['realloc_pos'])
country_df['realloc_neg'] = np.abs(country_df['realloc_neg'])
country_df['realloc'] = country_df['realloc_pos']

country_df['realloc_percent'] = (country_df['realloc'] / country_df['labor']) * 100

income_rank = pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
country_df = country_df.join(income_rank)

# %% Reallocation of production, nominal differences

print('Plotting production reallocation in nominal differences')

country_df.sort_values('realloc', inplace=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_df.index.get_level_values(0)
       , country_df.realloc / 1e6
       # ,bottom = country_dist_df.realloc_neg
       , label='Net reallocation of workers',
       # color=colors
       )

ax.set_xticklabels(['']
                   , rotation=75
                   # , ha='right'
                   # , rotation_mode='anchor'
                   , fontsize=19)
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('Million workers',
              fontsize=20)

# handles = []
# for ind in indicators_sorted:
# handles = [mpatches.Patch(color=palette[ind], label=country_dist_df.group_label.drop_duplicates().to_list()[ind]) for ind,group in enumerate(country_dist_df.group_code.drop_duplicates().to_list())]
# legend = ax1.legend(handles=handles,
#           fontsize=20,
#           # title='Greensourcing possibility',
#           loc='lower right')
# ax1.grid(visible=False)


leg = ax.legend(fontsize=20, loc='upper left')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
ax.set_yscale('log')
max_lim = country_df['realloc'].max() / 1e6
ax.set_ylim(0, max_lim + 100)

ax.bar_label(ax.containers[0],
             labels=country_df.index.get_level_values(0),
             rotation=90,
             label_type='edge',
             padding=2, zorder=10)

plt.show()

# %% Reallocation of labor, percentage changes - FIG 9

print('Plotting workforce reallocation in percentages')

country_df.sort_values('realloc_percent', inplace=True)

fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

palette = [sns.color_palette()[i] for i in [3,0,2]]
colors = [palette[ind] for ind in country_df.income_code]

ax1=ax.twinx()

ax.bar(country_df.index.get_level_values(0)
       , country_df.realloc_percent
       # ,bottom = country_dist_df.realloc_neg
       # , label='Net reallocation (%)'
       , color=colors
       )

ax.set_xticklabels(['']
                   , rotation=75
                   # , ha='right'
                   # , rotation_mode='anchor'
                   , fontsize=19)
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('% of national labor force',
              fontsize=25)

handles = []
# for ind in country_df.income_code.drop_duplicates().to_list():
handles = [mpatches.Patch(color=palette[ind], 
                          label=country_df[country_df.income_code == ind].income_label.drop_duplicates().to_list()[0]
                          ) 
           for ind in country_df.income_code.drop_duplicates().to_list()]
legend = ax1.legend(handles=handles,
          fontsize=20,
          # title='Greensourcing possibility',
           loc='upper left'
          )
ax1.grid(visible=False)
ax1.set_yticks([])

# leg = ax.legend(fontsize=20, loc='upper left')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-20,5)
max_lim = country_df['realloc_percent'].max()
ax.set_ylim(0, max_lim + 1)

ax.bar_label(ax.containers[0],
             labels=country_df.index.get_level_values(0),
             rotation=90,
             label_type='edge',
             padding=3, zorder=10, fontsize=15)
# plt.savefig('../tax_model/eps_figures_for_ralph_pres/within_country_disruption.eps',format='eps')
plt.show()

# # SAve for KIS
# plt.savefig('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/labor_realloc.eps',format='eps')
# keep = country_df[['realloc_percent']].copy()
# keep.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/labor_realloc.csv')


# %% Reallocation with constant prices - REAL OUTPUT PLOTS

# %% Calculate total output as q_hat * cons_b + m_hat * iot_b`
# Compute _hat_sol (real changes)
price_hat = sol_all[y].res.price_hat

cons_adj = pd.merge(
    sol_all[y].cons.new.reset_index(),
    price_hat.reset_index(),
    'left',
    left_on=['row_country', 'row_sector'],
    right_on=['country', 'sector'],
)[
    ['row_country', 'row_sector', 'col_country', 'new', 'price_hat']
].set_index(['row_country', 'row_sector', 'col_country'])
cons_adj['cons'] = cons_adj['new']/cons_adj['price_hat']
cons_adj = cons_adj[['cons']].groupby(level=[0,1]).sum()

iot_adj = pd.merge(
    sol_all[y].iot.new.reset_index(),
    price_hat.reset_index(),
    'left',
    left_on=['row_country', 'row_sector'],
    right_on=['country', 'sector'],
)[
    ['row_country', 'row_sector', 'col_country', 'col_sector', 'new', 'price_hat']
].set_index(['row_country', 'row_sector', 'col_country', 'col_sector'])
iot_adj['iot'] = iot_adj['new']/iot_adj['price_hat']
iot_adj = iot_adj[['iot']].groupby(level=[0,1]).sum()

output_adj = pd.DataFrame(
    cons_adj.cons + iot_adj.iot, columns=['new_adj']
).rename_axis(['country', 'sector']).join(sol_all[y].output)


#%% Production reallocation - Sector-wise

print('Computing production reallocation sector-wise')

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
    temp = output_adj.xs(sector,level=1).new_adj-output_adj.xs(sector,level=1).value
    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp>0].sum())
    sector_realloc_neg.append(temp[temp<0].sum())
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['output'] = output_adj.groupby(level=1).sum().value.values
sector_dist_df['output_new'] = output_adj.groupby(level=1).sum().new_adj.values
sector_dist_df['realloc_pos'] = sector_realloc_pos
sector_dist_df['realloc_neg'] = sector_realloc_neg
sector_dist_df['change'] = sector_change

sector_dist_df['realloc_pos'] = np.abs(sector_dist_df['realloc_pos'])
sector_dist_df['realloc_neg'] = np.abs(sector_dist_df['realloc_neg'])
sector_dist_df['realloc'] = sector_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
sector_dist_df['realloc'] = sector_dist_df['realloc'] * np.sign(sector_dist_df['change'])
sector_dist_df['change_tot_nom'] = (sector_dist_df['change']+sector_dist_df['realloc'])
sector_dist_df['realloc_share_nom'] = (sector_dist_df['realloc']/sector_dist_df['change_tot_nom']) * np.sign(sector_dist_df['change'])

sector_dist_df['realloc_percent'] = (sector_dist_df['realloc']/sector_dist_df['output'])*100
sector_dist_df['change_percent'] = (sector_dist_df['change']/sector_dist_df['output'])*100
sector_dist_df['change_tot'] = (sector_dist_df['change_percent']+sector_dist_df['realloc_percent'])
sector_dist_df['realloc_share_neg'] = (sector_dist_df['realloc_percent']/sector_dist_df['change_tot']) * np.sign(sector_dist_df['change'])

#%% Production reallocation, nominal differences

print('Plotting production reallocation in nominal differences')

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
            ,label='Net change of output',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc/1e6
            ,bottom = sector_use.change/1e6
            ,label='Reallocated output',
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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a sector \nacross countries for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(27,-1.25),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot_nom'].max()/1e6
min_lim = sector_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.3,max_lim+0.3)

plt.show()

#%% Production reallocation, % changes - FIG 7

print('Plotting production reallocation in percentages')

sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg', 'change_tot']].copy()
sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
sector_pos.sort_values('change_percent', ascending = True, inplace = True)
sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
sector_neg2.sort_values('change_percent',ascending = True, inplace = True)

sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of output (%)',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc_percent
            ,bottom = sector_use.change_percent
            ,label='Reallocated output (%)',
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
ax.set_ylabel('Percent change and reallocation', fontsize = 20)

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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a sector \nacross countries for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-30),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot'].max()
min_lim = sector_dist_df['change_tot'].min()
ax.set_ylim(min_lim-5,max_lim+10)

plt.show()
# # SAve for KIS
plt.savefig('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/sector_realloc.eps',format='eps')
keep = sector_use[['industry', 'change_percent', 'realloc_percent', 'change_tot']].copy()
keep.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/sector_realloc.csv')

#%% Country specific reallocation of production

print('Computing production reallocation country-wise')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_change = []
country_realloc_pos = []
country_realloc_neg = []
for country in country_list:
    temp = output_adj.xs(country,level=0).new_adj-output_adj.xs(country,level=0).value
    country_change.append(temp.sum())
    country_realloc_pos.append(temp[temp>0].sum())
    country_realloc_neg.append(temp[temp<0].sum())

country_dist_df = pd.DataFrame(index=country_list)
country_dist_df['output'] = output_adj.groupby(level=0).sum().value.values
country_dist_df['output_new'] = output_adj.groupby(level=0).sum().new_adj.values
country_dist_df['realloc_pos'] = country_realloc_pos
country_dist_df['realloc_neg'] = country_realloc_neg
country_dist_df['change'] = country_change
country_dist_df['share_percent'] = (country_dist_df['output']/country_dist_df['output'].sum())*100
country_dist_df['share_new_percent'] = (country_dist_df['output_new']/country_dist_df['output_new'].sum())*100

country_dist_df['realloc_pos'] = np.abs(country_dist_df['realloc_pos'])
country_dist_df['realloc_neg'] = np.abs(country_dist_df['realloc_neg'])
country_dist_df['realloc'] = country_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
country_dist_df['realloc'] = country_dist_df['realloc'] * np.sign(country_dist_df['change'])
country_dist_df['change_tot_nom'] = (country_dist_df['change']+country_dist_df['realloc'])

country_dist_df['realloc_percent'] = (country_dist_df['realloc']/country_dist_df['output'])*100
country_dist_df['change_percent'] = (country_dist_df['change']/country_dist_df['output'])*100
country_dist_df['total_change'] = country_dist_df['realloc_percent'] + country_dist_df['change_percent']


#%% Reallocation of production, nominal differences

print('Plotting production reallocation in nominal differences')

country_dist_df.sort_values('change_tot_nom',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.change/1e6
            # ,bottom = country_dist_df.realloc_neg
            ,label='Net change of output',
            # color=colors
            )

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.realloc/1e6
            ,bottom = country_dist_df.change/1e6
            ,label='Reallocated output',
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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(41,-3),fontsize=25,zorder=10,backgroundcolor='w')

max_lim = country_dist_df['change_tot_nom'].max()/1e6
min_lim = country_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.3,max_lim+0.1)

plt.show()

#%% Reallocation of production, percentage changes - FIG 8

print('Plotting production reallocation in percentages')

country_dist_df.sort_values('change_percent',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.change_percent
            # ,bottom = country_dist_df.realloc_neg
            ,label='Net change of output (%)',
            # color=colors
            )

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.realloc_percent
            ,bottom = country_dist_df.change_percent
            ,label='Reallocated output (%)',
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
ax.set_ylabel('Percent change and reallocation',
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
# ax.set_ylim(-20,5)

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(41,-20),fontsize=25,zorder=10,backgroundcolor='w')


max_lim = country_dist_df['total_change'].max()
min_lim = country_dist_df['total_change'].min()
ax.set_ylim(min_lim-2,max_lim+2)

plt.show()
# # SAve for KIS
plt.savefig('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/country_realloc.eps',format='eps')
keep = country_dist_df[['change_percent', 'realloc_percent', 'total_change']].copy()
keep.to_csv('/Users/malemo/Dropbox/UZH/Green Logistics/Global Sustainability Index/Presentation/KIS1_plots/country_realloc.csv')


# %% Reallocation with constant quantities (price adjustment effect)

# %% Calculate total output as p_hat * cons_b + p_hat * iot_b
# Compute _hat_sol (real changes)
price_hat = sol_all[y].res.price_hat

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


#%% Production reallocation - Sector-wise

print('Computing production reallocation sector-wise')

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
    temp = output_adj.xs(sector,level=1).new_adj-output_adj.xs(sector,level=1).value
    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp>0].sum())
    sector_realloc_neg.append(temp[temp<0].sum())
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['output'] = output_adj.groupby(level=1).sum().value.values
sector_dist_df['output_new'] = output_adj.groupby(level=1).sum().new_adj.values
sector_dist_df['realloc_pos'] = sector_realloc_pos
sector_dist_df['realloc_neg'] = sector_realloc_neg
sector_dist_df['change'] = sector_change

sector_dist_df['realloc_pos'] = np.abs(sector_dist_df['realloc_pos'])
sector_dist_df['realloc_neg'] = np.abs(sector_dist_df['realloc_neg'])
sector_dist_df['realloc'] = sector_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
sector_dist_df['realloc'] = sector_dist_df['realloc'] * np.sign(sector_dist_df['change'])
sector_dist_df['change_tot_nom'] = (sector_dist_df['change']+sector_dist_df['realloc'])
sector_dist_df['realloc_share_nom'] = (sector_dist_df['realloc']/sector_dist_df['change_tot_nom']) * np.sign(sector_dist_df['change'])

sector_dist_df['realloc_percent'] = (sector_dist_df['realloc']/sector_dist_df['output'])*100
sector_dist_df['change_percent'] = (sector_dist_df['change']/sector_dist_df['output'])*100
sector_dist_df['change_tot'] = (sector_dist_df['change_percent']+sector_dist_df['realloc_percent'])
sector_dist_df['realloc_share_neg'] = (sector_dist_df['realloc_percent']/sector_dist_df['change_tot']) * np.sign(sector_dist_df['change'])

#%% Production reallocation, nominal differences

print('Plotting production reallocation in nominal differences')

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
            ,label='Net change in prices',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc/1e6
            ,bottom = sector_use.change/1e6
            ,label='Price reallocation',
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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a sector \nacross countries for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-0.2),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot_nom'].max()/1e6
min_lim = sector_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.2,max_lim+0.2)

plt.show()

#%% Production reallocation, % changes

print('Plotting production reallocation in percentages')

sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg', 'change_tot']].copy()
sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
sector_pos.sort_values('change_percent', ascending = True, inplace = True)
sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
sector_neg2.sort_values('change_percent',ascending = True, inplace = True)

sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change in prices (%)',
            # color=colors
            )

ax.bar(sector_use.industry
            ,sector_use.realloc_percent
            ,bottom = sector_use.change_percent
            ,label='Price reallocation (%)',
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
ax.set_ylabel('Percent change and reallocation', fontsize = 20)

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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a sector \nacross countries for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-30),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

max_lim = sector_dist_df['change_tot'].max()
min_lim = sector_dist_df['change_tot'].min()
ax.set_ylim(min_lim-3,max_lim+2.5)

plt.show()

#%% Country specific reallocation of production

print('Computing production reallocation country-wise')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_change = []
country_realloc_pos = []
country_realloc_neg = []
for country in country_list:
    temp = output_adj.xs(country,level=0).new_adj-output_adj.xs(country,level=0).value
    country_change.append(temp.sum())
    country_realloc_pos.append(temp[temp>0].sum())
    country_realloc_neg.append(temp[temp<0].sum())

country_dist_df = pd.DataFrame(index=country_list)
country_dist_df['output'] = output_adj.groupby(level=0).sum().value.values
country_dist_df['output_new'] = output_adj.groupby(level=0).sum().new_adj.values
country_dist_df['realloc_pos'] = country_realloc_pos
country_dist_df['realloc_neg'] = country_realloc_neg
country_dist_df['change'] = country_change
country_dist_df['share_percent'] = (country_dist_df['output']/country_dist_df['output'].sum())*100
country_dist_df['share_new_percent'] = (country_dist_df['output_new']/country_dist_df['output_new'].sum())*100

country_dist_df['realloc_pos'] = np.abs(country_dist_df['realloc_pos'])
country_dist_df['realloc_neg'] = np.abs(country_dist_df['realloc_neg'])
country_dist_df['realloc'] = country_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
country_dist_df['realloc'] = country_dist_df['realloc'] * np.sign(country_dist_df['change'])
country_dist_df['change_tot_nom'] = (country_dist_df['change']+country_dist_df['realloc'])

country_dist_df['realloc_percent'] = (country_dist_df['realloc']/country_dist_df['output'])*100
country_dist_df['change_percent'] = (country_dist_df['change']/country_dist_df['output'])*100
country_dist_df['total_change'] = country_dist_df['realloc_percent'] + country_dist_df['change_percent']


#%% Reallocation of production, nominal differences

print('Plotting production reallocation in nominal differences')

country_dist_df.sort_values('change_tot_nom',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.change/1e6
            # ,bottom = country_dist_df.realloc_neg
            ,label='Net change of prices',
            # color=colors
            )

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.realloc/1e6
            ,bottom = country_dist_df.change/1e6
            ,label='Prices reallocation',
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

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(41,-3),fontsize=25,zorder=10,backgroundcolor='w')

max_lim = country_dist_df['change_tot_nom'].max()/1e6
min_lim = country_dist_df['change_tot_nom'].min()/1e6
ax.set_ylim(min_lim-0.15,max_lim+0.15)

plt.show()

#%% Reallocation of production, percentage changes

print('Plotting production reallocation in percentages')

country_dist_df.sort_values('total_change',inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.change_percent
            # ,bottom = country_dist_df.realloc_neg
            ,label='Net change of prices (%)',
            # color=colors
            )

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.realloc_percent
            ,bottom = country_dist_df.change_percent
            ,label='Prices reallocation (%)',
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
ax.set_ylabel('Percent change and reallocation',
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
# ax.set_ylim(-20,5)

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

total_output_net_decrease = output_adj.value.sum() - output_adj.new_adj.sum()
total_output = output_adj.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross (real) output \nwould be reallocated within a country \nacross sectors for a net reduction \nof (real) output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(41,-20),fontsize=25,zorder=10,backgroundcolor='w')


max_lim = country_dist_df['total_change'].max()
min_lim = country_dist_df['total_change'].min()
ax.set_ylim(min_lim-1,max_lim+1)

plt.show()


#%% Relate welfare change to wage change for each country
print('Computing inequalities in terms of GDP change')

# Construct GDP per capita and welfare change
gdp = sol_all[y].va.groupby(level=0).sum()
gdp['price_index'] = price_index
gdp['wage_hat'] = (wage.hat.to_numpy()-1)*100

gdp = gdp.join(labor_year).rename(columns={year:'labor'})
gdp['per_capita'] = (gdp.value / gdp.labor)*1e3

gdp['utility_percent_change'] = (sol_all[y].utility.values-1)*100

# Format regions for kernel density
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')
gdp = gdp.join(country_map)

gdp.loc['TWN','Continent'] = 'Asia'
gdp.loc['ROW','Continent'] = 'Africa'
gdp.loc['AUS','Continent'] = 'Asia'
gdp.loc['NZL','Continent'] = 'Asia'
gdp.loc['CRI','Continent'] = 'South America'
gdp.loc['RUS','Continent'] = 'Asia'
gdp.loc['SAU','Continent'] = 'Africa'
gdp.loc['CAN','labor'] = gdp.loc['CAN','labor']*6
gdp.loc['MEX','labor'] = gdp.loc['MEX','labor']*2

#%% Plot - FIG 10
print('Plotting inequalities')

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }
colors = [continent_colors[gdp.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

# ax.scatter(gdp.per_capita,gdp.utility_percent_change,marker='x',lw=2,s=50)
#
# ax.set_xlabel('GDP per workforce (Thousands $)', fontsize = 20)
# ax.set_ylabel('Welfare change (%)', fontsize = 20)
ax.scatter(gdp.wage_hat,gdp.utility_percent_change,marker='x',lw=2,s=50)

ax.set_xlabel('GDPpc change (%)', fontsize = 20)
ax.set_ylabel('Welfare change (%)', fontsize = 20)

# sns.kdeplot(data=gdp,
#                 x='per_capita',
#                 y="utility_percent_change",
#                 hue = 'Continent',
#                 fill = True,
#                 alpha = 0.25,
#                 # height=10,
#                 # ratio=5,
#                 # bw_adjust=0.7,
#                 weights = 'labor',
#                 # legend=False,
#                 levels = 2,
#                 palette = palette,
#                 # common_norm = False,
#                 shade=True,
#                 thresh = 0.12,
#                 # dropna=True,
#                 # fill = False,
#                 # alpha=0.6,
#                 # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
#                 ax = ax
#                 )

# sns.move_legend(ax, "lower right")

coeffs_fit = np.polyfit(gdp.wage_hat,
                  gdp.utility_percent_change,
                  deg = 1,
                  # w=own_trade.labor
                  )

x_lims = (-12,4)
ax.set_xlim(*x_lims)
y_lims = (-4,2)
ax.set_ylim(*y_lims)

x_vals = np.arange(x_lims[0],x_lims[1]+1)
y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')

ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')

ax.set_xlim(-12,4)
# ax.set_ylim(-6.5,2.5)             # For kernel density
ax.set_ylim(-4,2)


# texts = [plt.text(gdp.per_capita.loc[country], gdp.utility_percent_change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]     # For kernel density
texts = [plt.text(gdp.wage_hat.loc[country], gdp.utility_percent_change.loc[country], country,size=15) for i,country in enumerate(country_list)]

# adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))

plt.show()

# %% Calculating effective tax rate
# Compute aggregate spending, and spending weighted by carbon intensity
m_df = sol_all[y].iot.groupby(level=[0, 1, 2]).sum().new
s_df = pd.merge(
    sol_all[y].cons.new,
    m_df,
    right_index=True,
    left_index=True,
    suffixes=('_cons', '_m')
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
s_df['spending_carb'] = s_df['spending'] * s_df['value'] * carb_cost  # Tons of CO2 * millions of dollars / Tons

# Aggregate at the consuming country level
sagg_df = s_df.groupby(['col_country']).sum()[['spending', 'spending_carb']]
sagg_df['tax'] = sagg_df['spending_carb'] / sagg_df['spending'] * 100

# Summarise
sagg_df['tax'].describe()

# Bar Plot
sagg_df.reset_index(inplace=True)
sagg_df.sort_values('tax', ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

ax.bar(sagg_df.col_country
       , sagg_df.tax
       # ,bottom = sector_dist_df.realloc_neg
       , label='Effective tax rate on spending',
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
ax.set_ylabel('% of total spending', fontsize=20)

ax.bar_label(ax.containers[0],
             labels=sagg_df.col_country,
             rotation=90,
             label_type='edge',
             padding=2,
             zorder=10)

ax.set_ylim(0, 10)
plt.show()

# %%  Effective transfers in order to equalize welfare changes
print('Plotting adjustments needed for a fair tax')

gdp = sol_all[y].va.groupby(level=0).sum()
gdp['price_index'] = price_index

gdp = gdp.join(labor_year).rename(columns={year: 'labor'})
gdp['per_capita'] = (gdp.value / gdp.labor)

gdp['change_real'] = ((gdp.new / gdp.value) / (gdp.price_index))

# gdp['new_adjusted'] = gdp['change_real']*gdp['value']
gdp['new_adjusted'] = gdp['new']

gdp_mean_change = gdp['new_adjusted'].sum() / gdp['value'].sum()

gdp['new_if_average_change_adjusted'] = gdp['value'] * gdp_mean_change

gdp['contribution'] = gdp['new_adjusted'] - gdp['new_if_average_change_adjusted']

gdp['wage_change_for_equality'] = -(gdp['contribution'] / gdp.labor) * 1e6

gdp['relative_change'] = (gdp['wage_change_for_equality'] / (gdp.per_capita * 1e6)) * 100

gdp.sort_values('wage_change_for_equality', inplace=True)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

ax1 = ax.twinx()

ax.bar(gdp.index.get_level_values(0), gdp['wage_change_for_equality'])

ax.set_xticklabels([''])
ax.tick_params(axis='y', colors=sns.color_palette()[0], labelsize=20)
ax.set_ylabel('Contribution per worker ($)', color=sns.color_palette()[0], fontsize=30)

ax.bar_label(ax.containers[0],
             labels=gdp.index.get_level_values(0),
             rotation=90,
             label_type='edge',
             padding=2, zorder=10)

ax1.scatter(gdp.index.get_level_values(0), gdp['relative_change'], color=sns.color_palette()[1])

ax1.grid(visible=False)
ax.margins(x=0.01)
ax.set_ylim(-6500, 6500)
ax1.set_ylim(-6.5, 6.5)

ax1.tick_params(axis='y', colors=sns.color_palette()[1], labelsize=20)
ax1.set_ylabel('Contribution per worker (% of GDPpc)', color=sns.color_palette()[1], fontsize=30)

plt.show()

# Calculate average transfer
pos_total = gdp.loc[gdp['contribution'] > 0, 'contribution'].sum()
posL_total = gdp.loc[gdp['contribution'] > 0, 'labor'].sum()
neg_total = gdp.loc[gdp['contribution'] <= 0, 'contribution'].sum()
negL_total = gdp.loc[gdp['contribution'] <= 0, 'labor'].sum()

print('Positive transfer per capita:', pos_total / posL_total * 1e6)
print('Negative transfer per capita:', neg_total / negL_total * 1e6)

# %% Plot 3 stat: Complete reallocation (across both origins and sectors)
calc = output
temp = calc.new - calc.value
change = temp.sum()
realloc_pos = np.abs(temp[temp > 0].sum())
realloc_neg = np.abs(temp[temp < 0].sum())
total_value = calc.value.sum()
total_new = calc.new.sum()
realloc = np.minimum(realloc_pos, realloc_neg)
realloc_percent = (realloc / total_value) * 100
change_percent = (change / total_value) * 100
total_change = realloc_percent + change_percent

print(
    f'Overall {round(realloc_percent, 2)} percent of gross output would be reallocated across countries and sectors for a net reduction of output of {round(change_percent, 2)}%')

# %% Within country sectoral reallocation - get examples
# Construct dataframe
output = sol_all[y].output

output['diff'] = output['new'] - output['value']
output.reset_index(inplace=True)


#%% Test to approximate changes in terms of trade
# Import prices are aggregate consumer price index = price_index
# Export prices are price deflator on total exports - approximate its change as a
# weighted average of sectoral price changes (factory based since tau doesn't move)
# where weights are export share in the baseline (i.e. ignoring quantity adjustments)

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

#%% Correlate ToT change and welfare change
ToT_df['utility_change'] = (sol_all[y].utility.values-1)*100

# Format regions for kernel density
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')
ToT_df = ToT_df.join(country_map)

ToT_df.loc['TWN','Continent'] = 'Asia'
ToT_df.loc['ROW','Continent'] = 'Africa'
ToT_df.loc['AUS','Continent'] = 'Asia'
ToT_df.loc['NZL','Continent'] = 'Asia'
ToT_df.loc['CRI','Continent'] = 'South America'
ToT_df.loc['RUS','Continent'] = 'Asia'
ToT_df.loc['SAU','Continent'] = 'Africa'

#%% Plot
print('Plotting inequalities')

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }
colors = [continent_colors[ToT_df.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

ax.scatter(ToT_df.ToT_hat_percent,ToT_df.utility_change,marker='x',lw=2,s=50)

ax.set_xlabel('ToT change (%)', fontsize = 20)
ax.set_ylabel('Welfare change (%)', fontsize = 20)

# sns.kdeplot(data=ToT_df,
#                 x='per_capita',
#                 y="utility_percent_change",
#                 hue = 'Continent',
#                 fill = True,
#                 alpha = 0.25,
#                 # height=10,
#                 # ratio=5,
#                 # bw_adjust=0.7,
#                 weights = 'labor',
#                 # legend=False,
#                 levels = 2,
#                 palette = palette,
#                 # common_norm = False,
#                 shade=True,
#                 thresh = 0.12,
#                 # dropna=True,
#                 # fill = False,
#                 # alpha=0.6,
#                 # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
#                 ax = ax
#                 )

# sns.move_legend(ax, "lower right")

coeffs_fit = np.polyfit(ToT_df.ToT_hat_percent,
                  ToT_df.utility_change,
                  deg = 1,
                  # w=own_trade.labor
                  )

x_lims = (-4,2)
ax.set_xlim(*x_lims)
y_lims = (-4,1)
ax.set_ylim(*y_lims)

# x_vals = np.arange(x_lims[0],x_lims[1]+1)
# y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
# ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')
#
# ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')
#
# ax.set_xlim(-12,4)
# # ax.set_ylim(-6.5,2.5)             # For kernel density
# ax.set_ylim(-4,2)
#

# texts = [plt.text(ToT_df.per_capita.loc[country], ToT_df.utility_percent_change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]     # For kernel density
texts = [plt.text(ToT_df.ToT_hat_percent.loc[country], ToT_df.utility_change.loc[country], country,size=15) for i,country in enumerate(country_list)]

# adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

plt.show()
