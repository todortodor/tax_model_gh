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

#%% Intro plots (data pres)

#%% GHG distrib over time by sector of use - Climate watch data

print('Plotting GHG distribution by categories')

years = [y for y in range(1995,2019)]

emissions_baseline = [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(year)+'/prod_CO2_WLD_with_agri_agri_ind_proc_fug_'+str(year)+'.csv').value.sum() for year in years]

ghg = pd.read_csv('data/climate_watch/ghg_EDGAR_ip.csv').set_index(['sector','year'])

datas = [ 'Electricity/Heat',
        'Manufacturing/Construction','Transportation','Agriculture','Forest fires',
         'Fugitive Emissions', 'Industrial Processes','Other Fuel Combustion',
        'Waste']

labels = [ 'Energy (fuel)',
        'Industries (fuel)','Transports (fuel)','Agriculture (direct emissions) \nand related land-use change',
        'Forestry - Land change',  
         'Fugitive Emissions', 'Industrial Processes','Other Fuel Combustion',
        'Waste (direct emissions)']

to_plot = [ghg.loc[data].value for data in datas]

fig, ax = plt.subplots(figsize=(16,8),constrained_layout=True)

palette2 = [sns.color_palette()[i] for i in [2,1,3,0]]
palette2[0] = (0.4333333333333333, 0.5588235294117647, 0.40784313725490196)
palette = [*sns.color_palette("dark:salmon_r",n_colors=6)[::-1][:5] , *palette2]
palette[3] , palette[5] = palette[5] , palette[3]
palette[4] , palette[7] = palette[7] , palette[4]
palette[6] , palette[7] = palette[7] , palette[6]

stacks = ax.stackplot(years, 
             to_plot,
             labels = labels,
                colors = palette,zorder = -10,
                linewidth=0.5
             )

hatches = ['','','','','','','','//','//']
for stack, hatch in zip(stacks, hatches):
    stack.set_hatch(hatch)

plt.grid(alpha=0.4)

ax.plot(years,np.array(emissions_baseline)/1e3,ls='--',lw=6,color = sns.color_palette('Set2')[5],label = 'Data')

ax.plot(years, (-ghg.loc['Forestland']).value, ls = '--', lw=3, label = 'Forest carbon sinks \nNatural net zero target', color= 'g')

handles, labels = plt.gca().get_legend_handles_labels()
order = [9,0,1,2,3,4,5,6,7,8,10]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order]
          ,loc=(-0.60,-0.02),fontsize = 25) 
ax.margins(x=0)

ax.tick_params(axis='both', labelsize=20 )
# ax.set_title('Main greenhouse gases emissions (in Gt of CO2 equivalent)',fontsize=25,pad=15)

plt.show()

#%% GHG distrib over time by gas - Climate watch data

print('Plotting GHG distribution by gas')

carbon_prices = pd.read_csv('data/emissions_priced.csv')

ghg = pd.read_csv('data/climate_watch/ghg_EDGAR_gas.csv').set_index('gas')

datas = ['CO2_data', 'CO2', 'CH4_data', 'CH4', 'N2O_data', 'N2O',   'F-Gas']

to_plot = [ghg.loc[data].value for data in datas]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

palette = [sns.color_palette()[i] for i in [7,7,5,5,3,3,9]]

stacks = ax.stackplot(years, 
             *to_plot,
             # labels = labels,
             colors = palette, 
             zorder = -10,
             lw = 0.5
             )
hatches = [None,'/',None,'/',None,'/',None]
for stack, hatch in zip(stacks, hatches):
    stack.set_hatch(hatch)
    
plt.grid(alpha=0.4)

# phantom plots for legend. it's dirty !
legend_labels = ['Carbon dioxyde','Methane','Nitrous oxyde','F-Gas','Datas','Unaccounted for']
legend_colors = [sns.color_palette()[i] for i in [7,5,3,9,0,0]]
legend_hatches = [None]*5+['//']
stacks_legend = ax.stackplot([],[[]*24]*6,colors = legend_colors,labels = legend_labels)
for stack_legend, hatch in zip(stacks_legend, legend_hatches):
    stack_legend.set_hatch(hatch)
    
ax.plot(years, carbon_prices.value, ls = '--', lw=5, label = 'Emissions falling under some\ntype of carbon pricing scheme', color= 'r')    
    
ax.legend(loc='lower left'
           ,fontsize=17)


ax.margins(x=0)

ax.tick_params(axis='both', labelsize=20 )
ax.set_title('Main greenhouse gases emissions (in Gt of CO2 equivalent)',fontsize=25,pad=15)

plt.show()

#%% Pie charts distributions

print('Plotting pie charts of the distribution of emissions by sector/country')

y  = 2018
year = str(y)

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
sector_list = sector_map.industry.to_list()

countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country')
country_list = countries.country.to_list()

S = len(sector_list)
C = len(country_list)

cons, iot, output, va, co2_prod, co2_intensity = t.load_baseline(year)

sh = t.shares(cons, iot, output, va, co2_prod, co2_intensity)

sector_list_noD = []
for sector in sector_map.index.to_list():
    sector_list_noD.append(sector[1:])

years = [y for y in range(1995,2019)]

emissions_baseline = [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(year)+'/prod_CO2_WLD_with_agri_agri_ind_proc_fug_'+str(year)+'.csv').value.sum() for year in years]


fig, ax = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
# ,constrained_layout=True
nbr_labels = 10
fontsize = 10
fontsize_title = 20
lab_distance = 1.04
center = (0,0)
radius = 1

data = co2_prod.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]
colors_emissions = sns.diverging_palette(145, 50, s=80,l=40)
colors_output =sns.diverging_palette(220, 20, s=80,l=40)

colors_emissions = sns.diverging_palette(200, 130, s=80,l=40)
colors_output =sns.diverging_palette(60, 20, s=80,l=40)

ax[0,0].pie(x = data,
        labels = labels,
        explode=np.ones(len(countries.country))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)

ax[0,0].set_title('Emissions distribution',fontsize=fontsize_title,y=0.8,x=1.25)

data = output.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]

ax[1,0].pie(x = data,
        labels = labels,
        explode=np.ones(len(countries.country))-0.98,
        colors=colors_output,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


ax[1,0].set_title('Gross output distribution',fontsize=fontsize_title,y=0.8,x=1.25)

data = co2_prod.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

ax[0,1].pie(x = data,
        labels = labels,
        explode=np.ones(len(sector_map.industry))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


# ax[0,1].set_title('Emission output distribution',fontsize=fontsize_title)

data = output.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

ax[1,1].pie(x = data,
        labels = labels,
        explode=np.ones(len(sector_map.industry))-0.98,
        colors=colors_output,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


# ax[1,1].set_title('Gross output distribution',fontsize=fontsize_title)

# plt.tight_layout()

plt.show()

#%% Existing carbon pricing

print('Plotting existing carbon tax / ETS system')

carbon_prices_all = pd.read_csv('data/weighted_average_prices_rest0.csv')
carbon_prices_all = carbon_prices_all.replace('combined','Combined').set_index(['instrument','year'])
carbon_prices = pd.read_csv('data/weighted_average_prices.csv')
carbon_prices = carbon_prices.replace('combined','Mean').set_index(['instrument','year'])

fig, ax = plt.subplots(figsize=(12,8))

for data in carbon_prices.index.get_level_values(0).drop_duplicates():
    ax.plot(years[-len(carbon_prices.loc[data].price):],carbon_prices.loc[data].price, lw=3, label = data)

ax.legend()
ax.set_ylabel('Dollars per ton of carbon',fontsize = 20)
plt.title('Average carbon price on emissions that fall under a pricing scheme',fontsize=20)

plt.show()

fig, ax = plt.subplots(figsize=(12,8))

for data in carbon_prices_all.index.get_level_values(0).drop_duplicates():
    ax.plot(years[-len(carbon_prices_all.loc[data].price):],carbon_prices_all.loc[data].price, lw=3, label = data)

ax.legend()
ax.set_ylabel('Dollars per ton of carbon',fontsize = 20)
plt.title('Average world carbon price, averaged on all emissions',fontsize=20)

plt.show()

#%% Agriculture emissions

print('Plotting Agriculture emissions details')

emissions_agri = pd.read_csv('data/agriculture_emissions_subsectors.csv').sort_values('category')
palette = []*len(emissions_agri)
palette[:4] = sns.dark_palette("grey", n_colors=5)[1:]#[::-1]
palette[4:7] = sns.dark_palette("green", n_colors=5)[1:][::-1]
palette[8:] = sns.dark_palette("red", n_colors=6)[1:][::-1]

fig, ax = plt.subplots(figsize=(10,10))

ax.pie(x = emissions_agri.total_CO2eq,
        labels = emissions_agri.Item,
        explode=np.ones(len(emissions_agri.Item))-0.98,
        colors=palette,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':20},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)

plt.show()

#%% Baseline 2018 - range of carbon costs plots

print('Setting parameters for run')

y = 2018
year = str(y)
dir_num = 8
path = 'results/'+year+'_'+str(dir_num)

#%% Efficacy of carbon tax

print('Plotting efficacy of carbon tax in terms of emissions reduction')

runs = pd.read_csv(path+'/runs')
runs['emissions'] = runs['emissions']/1e3
runs['carb_cost'] = runs['carb_cost']*1e6
runs_low_carb_cost = runs[runs['carb_cost'] <= 1e3]

fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:green'

ax1.set_xlabel('Carbon tax (dollar / ton of CO2)',size = 30)
ax1.set_xlim((runs_low_carb_cost.carb_cost).min(),(runs_low_carb_cost.carb_cost).max())
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Global emissions (Gt)', color=color,size = 30)
ax1.plot((runs_low_carb_cost.carb_cost),(runs_low_carb_cost.emissions), color=color,lw=5)
ax1.tick_params(axis='y', labelsize = 20)

y_100 = runs_low_carb_cost.iloc[np.argmin(np.abs(runs_low_carb_cost.carb_cost-100))].emissions
y_0 = runs_low_carb_cost.iloc[0].emissions

ax1.vlines(x=100,
           ymin=0,
           ymax=y_100,
           lw=3,
           ls = '--',
           color = color)

ax1.hlines(y=y_100,
           xmin=0,
           xmax=100,
           lw=3,
           ls = '--',
           color = color)

ax1.annotate('100',xy=(100,0), xytext=(-20,-20), textcoords='offset points',color=color)
ax1.annotate(str(y_100.round(1)),
             xy=(0,y_100),
             xytext=(-37,-10),
             textcoords='offset points',color=color)

ax1.annotate(str(y_0.round(1)),
              xy=(0,y_0),
              xytext=(-37,-6),
              textcoords='offset points',color=color)

ax1.annotate("$100/Ton tax would reduce emissions by "+str(((y_0-y_100)*100/y_0).round(1))+"%",
            xy=(100, y_100), xycoords='data',
            xytext=(100+5, y_100+4),
            textcoords='data',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3",color= 'black'),
            bbox=dict(boxstyle="round", fc="w")
            )

ax1.margins(y=0)

ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
ax1.set_yticklabels(['0', '5', '10', '15', '20', '25', '30', '35', '40', '45'])

fig.tight_layout()

plt.show()

#%% Load data for effect on GDP and welfare and Price indexes and Share output traded

print('Loading welfare, GDP cost, price index changes, share output traded changes corresponding to a carbon tax')

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



traded_new = []
traded_share_new = []
gross_output_new = []
gdp_new = []
utility = []
emissions = []
gdp = []
# dist = []
countries = country_list
price_index_l = {}

for country in countries:
    price_index_l[country] = []

carb_cost_l = np.linspace(0, 1e-3, 101)

for carb_cost in tqdm(carb_cost_l):
    run = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]
    # utility.append(run.utility)
    emissions.append(run.emissions)

    sigma = run.sigma
    eta = run.eta
    num = run.num
    carb_cost = run.carb_cost

    # res = pd.read_csv(run.path).set_index(['country','sector'])

    cons, iot, output, va, co2_prod, price_index, utility_countries = t.sol_from_loaded_data(carb_cost, run, cons_b, iot_b, output_b, va_b,
                                                                          co2_prod_b, sh)

    gross_output_new.append(cons.new.to_numpy().sum() + iot.new.to_numpy().sum())
    traded_new.append(
        (cons.new.to_numpy() * cons_traded_unit_np).sum() + (iot.new.to_numpy() * iot_traded_unit_np).sum())
    traded_share_new.append(traded_new[-1] / gross_output_new[-1])
    gdp_new.append(va.new.sum())
    gdp.append(va.value.sum())
    utility.append(utility_countries.new.mean())

    for country in countries:
        price_index_l[country].append(price_index[country_list.index(country)])

#%% Plot

print('Plotting welfare and GDP cost corresponding to a carbon tax')


fig, ax = plt.subplots(2,2,figsize=(12,8))

color = 'g'

# Upper left - Emissions
ax[0,0].plot(np.array(carb_cost_l)*1e6,np.array(emissions)/1e3,lw=4,color=color)
ax[0,0].legend(['Global emissions (Gt)'])
ax[0,0].set_xlabel('')
ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,0].set_xlim(0,1000)

y_100 = np.array(emissions)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]/1e3
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,0].vlines(x=100,
            ymin=0,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].margins(y=0)

ax[0,0].annotate(str(y_100.round(0)),
             xy=(0,y_100),
             xytext=(-37,-5),
             textcoords='offset points',color=color)

ax[0,0].set_ylim(0,np.array(emissions).max()/1e3+0.5)

ax[0,0].set_yticks([0, 10, 20, 30, 40])
ax[0,0].set_yticklabels(['0','10','20','30','40'])

# Upper right - GDP
gdp_covid = np.array(gdp_new).max()*0.955
tax_covid = carb_cost_l[np.argmin(np.abs(np.array(gdp_new) - gdp_covid))]

ax[0,1].plot(np.array(carb_cost_l)*1e6,np.array(gdp_new)/1e6,lw=4)
ax[0,1].set_xlabel('')
ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,1].set_xlim(0,1000)
ax[0,1].hlines(y=gdp_covid/1e6,linestyle=":",xmin=0,xmax=tax_covid*1e6,lw=3)
ax[0,1].legend(['GDP (thousand billion dollars)','GDP drop due to covid (scaled)'])
# ax[0,1].vlines(x=e_max,ymax=(GDP+DISU).max(), ymin=0, color=color, linestyle=":",lw=3)

y_100 = np.array(gdp_new)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]/1e6
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,1].vlines(x=100,
            ymin=np.array(gdp_new).min()/1e6,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].margins(y=0)

ax[0,1].annotate(str(y_100.round(1)),
              xy=(0,y_100),
              xytext=(-37,-5),
              textcoords='offset points',color=color)

ax[0,1].set_ylim(np.array(gdp_new).min()/1e6,np.array(gdp_new).max()/1e6+0.5)

# ax[0,1].set_yticks([79,80,81,82,83])
# ax[0,1].set_yticklabels(['79','80','81','','83'])

# gdp_covid = 0.955
# tax_covid = carb_cost_l[np.argmin(np.abs(np.array(gdp_new)/gdp_new[0] - gdp_covid))]

# ax[0,1].plot(np.array(carb_cost_l)*1e6,np.array(gdp_new)/gdp_new[0],lw=4)
# ax[0,1].set_xlabel('')
# ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# ax[0,1].set_xlim(0,1000)
# ax[0,1].hlines(y=gdp_covid,linestyle=":",xmin=0,xmax=tax_covid*1e6,lw=3)
# ax[0,1].legend(['GDP','GDP drop due to covid'])

# y_100 = (np.array(gdp_new)/gdp_new[0])[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]
# y_100 = (np.array(gdp_new))[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]

# ax[0,1].vlines(x=100,
#             ymin=(np.array(gdp_new)/gdp_new[0]).min(),
#             ymax=y_100,
#             lw=3,
#             ls = '--',
#             color = color)

# ax[0,1].hlines(y=y_100,
#             xmin=0,
#             xmax=100,
#             lw=3,
#             ls = '--',
#             color = color)

# ax[0,1].margins(y=0)

# ax[0,1].annotate(str(y_100.round(3)),
#              xy=(0,y_100),
#              xytext=(-50,-5),
#              textcoords='offset points',color=color)

# # ax[0,1].annotate(str(gdp_covid),
# #              xy=(0,gdp_covid),
# #              xytext=(-50,-5),
# #              textcoords='offset points', color='b', fontsize=15)

# ax[0,1].set_ylim((np.array(gdp_new)/gdp_new[0]).min(),(np.array(gdp_new)/gdp_new[0]).max()+0.005)

# Bottom left - Welfare
ax[1,0].plot(np.array(carb_cost_l)*1e6,utility,lw=4,color='r')
ax[1,0].legend(['Welfare from consumption'])
ax[1,0].set_xlabel('Carbon tax ($/ton of CO2)')
ax[1,0].set_xlim(0,1000)
# ax[1,0].set_ylim(min(utility),1.001)

y_100 = np.array(utility)[np.argmin(np.abs(np.array(carb_cost_l)*1e6-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[1,0].vlines(x=100,
            ymin=np.array(utility).min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].margins(y=0)

ax[1,0].set_ylim(np.array(utility).min(),1.005)

ax[1,0].annotate(str(y_100.round(3)),
              xy=(0,y_100),
              xytext=(-50,-10),
              textcoords='offset points',color=color)

ax[1,1].plot(np.array(carb_cost_l)*1e6,np.array(gdp_new)/gdp_new[0],lw=4)
ax[1,1].plot(np.array(carb_cost_l)*1e6,utility,lw=4,color='r')
ax[1,1].plot(np.array(carb_cost_l)*1e6,np.array(emissions)/emissions[0],lw=4,color='g')
ax[1,1].legend(['GDP','Welfare','Emissions'])
ax[1,1].set_xlabel('Carbon tax (dollar/ton of CO2)')
ax[1,1].set_xlim(0,1000)

plt.tight_layout()

plt.show()

#%% Effect on output share traded

print('Plotting share of output traded')

fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:blue'

ax1.set_xlabel('Carbon tax (dollar / ton of CO2)',size = 30)
ax1.set_xlim(0,1000)
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Share of output traded (%)', color=color,size = 30)
ax1.plot(np.array(carb_cost_l)*1e6,np.array(traded_share_new)*100, color=color,lw=5)
ax1.tick_params(axis='y', labelsize = 20)

# y_100 = (np.array(traded_share_new)/traded_share_new[0])[np.argmin(np.abs(np.array(carb_cost_l)*1e6 -100))]

# ax[1,0].vlines(x=100,
#             ymin=0.995,
#             ymax=y_100,
#             lw=3,
#             ls = '--',
#             color = color)
#
# ax[1,0].hlines(y=y_100,
#             xmin=0,
#             xmax=100,
#             lw=3,
#             ls = '--',
#             color = color)

ax1.margins(y=0)

# ax1.set_ylim((np.array(traded_share_new)*100).min()-0.05, (np.array(traded_share_new)*100).max() + 0.05)
ax1.set_ylim(12,15)

# ax[1,0].annotate(str(y_100.round(3)),
#               xy=(0,y_100),
#               xytext=(-50,-5),
#               textcoords='offset points',color=color)
#
# ax[1,0].set_yticks([1.00,1.01,1.02,1.03])
# ax[1,0].set_yticklabels(['', '1.01', '1.02', '1.03'])

plt.tight_layout
plt.show()

#%% Accounting for SCC

print('Plotting comparison between tax and SCC')

# Convert variables into relevant units
GDP_B = va.value.sum()/1e6     # Thousand billions $
emissions = runs.emissions/1e3
GDP = runs.utility*GDP_B    # Welfare expressed in dollar terms from GDP

fig , ax = plt.subplots(figsize=(12,8))
ax2 = ax.twinx()
#
bottom_left = 60
bottom_right = -10
height = 25

fontsize = 20
labelsize = 20

color1 = sns.color_palette()[0]
color2 = sns.color_palette()[3]
color3 = sns.color_palette()[4]

ax.tick_params(axis='x', which='major', labelsize=labelsize,color='grey')
ax.tick_params(axis='y', which='major', labelsize=labelsize,labelcolor=color1,color='grey')
ax.set_xlim(emissions[0], emissions.iloc[-1])
ax.set_ylim(bottom_left,bottom_left+height)
ax2.tick_params(axis='x', which='major', labelsize=labelsize,color='grey')
ax2.tick_params(axis='y', which='major', labelsize=labelsize,labelcolor=color2,color='grey')
ax.set_xlabel('Carbon emissions (Gt)',fontsize=fontsize)
ax.set_ylabel('Welfare (trillion $)',fontsize=fontsize)
ax2.set_ylabel('Welfare loss',fontsize=fontsize)
ax2.set_ylim(bottom_right,bottom_right+height)
ax.margins(y=0)
ax2.margins(y=0)

scc = 100/1e3
DISU = -scc*emissions
DISU = DISU #- DISU.max()

g_max = (GDP+DISU).max()
e_max = emissions[np.argmax(GDP+DISU)]
ax.hlines(y=(GDP+DISU).max(), xmax=e_max,xmin=emissions[0], linestyle=":",lw=4,color=color3)
ax.vlines(x=e_max,ymax=(GDP+DISU).max(), ymin=0, linestyle=":",lw=4,color=color3)

# ax.set_xticks([10.0,12.5,15.0,17.5,20,22.5,25.0,27.5])

ax3 = ax.twiny()
ax3.set_xlim(ax.get_xlim())
def tick_function(tick_locations):
    return [str((runs.carb_cost.iloc[np.argmin(np.abs(runs.emissions - tick*1e3))]*1e6).round(1)) for tick in tick_locations]
tick_locations = ax.get_xticks().tolist()
new_ticks = tick_function(tick_locations)
ax3.set_xticks(ax.get_xticks().tolist())
ax3.set_xticklabels(new_ticks,fontsize=labelsize,color='grey')
ax3.tick_params(axis='x', which='major',color='grey')
ax3.set_xlabel('Corresponding tax ($/Ton CO2)',fontsize=fontsize,color='grey',labelpad = 15)


ln1 = ax.plot(emissions,GDP,label = 'Welfare from consumption',lw=4,color=color1)
ln2 = ax.plot(emissions,GDP+DISU,lw=4,ls='-',label = 'Welfare net of damages',color = color3)
ln3 = ax2.plot(emissions,DISU,lw=4,ls='-',label = 'Welfare loss due to CO2 emissions',color=color2)
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs,fontsize = 17.5,loc='lower right')

# gridlines = ax.yaxis.get_gridlines()
# gridlines[ax.get_yticks().tolist().index(72.0)].set_color(color2)
# gridlines[ax.get_yticks().tolist().index(72.0)].set_linestyle('-.')
# gridlines[ax.get_yticks().tolist().index(72.0)].set_linewidth(1.5)

ax2.grid(visible=False)
ax3.grid(visible=False)

# ax.set_zorder(10)
# ax2.set_zorder(10)
# ax3.set_zorder(10)

plt.tight_layout()

plt.show()

#%% Consumer price index for a selection of countries

print('Plotting consumer price index changes for multiple countries')

# countries = ['DEU','CHN','USA','CHE']#,'FRA','CHE','CZE']
countries = country_list
# countries_to_label = ['DEU','CHN','USA','CHE']
countries_to_label = country_list

infl = {}
infl['DEU'] = 5.1
infl['CHN'] = 0.9
infl['USA'] = 7.9
infl['CHE'] = 2.2
# infl['AUS'] = 5 #not true, was just to plot

inf_l = [5.1,0.9,7.9,2.2]
carb_tax_eq_l = []
plot_inflation = False

if plot_inflation:
    for i,country in enumerate(countries):
        carb_tax_eq_l.append(carb_cost_l[np.argmin(np.abs((np.array(price_index_l[country])-1)*100-infl[country]))])

fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:blue'

ax1.set_xlabel('Carbon tax (dollar / ton of CO2)',size = 30)
ax1.set_xlim(0,1000)
# ax1.set_ylim(0,25)
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Consumer price index change (%)',size = 28)
for i,country in enumerate(countries):
    color=sns.color_palette()[i%10]
    if country in countries_to_label:
        ax1.plot(np.array(carb_cost_l)*1e6,(np.array(price_index_l[country])-1)*100,lw=2,label=country,color=color)
    # else:
    #     ax1.plot(np.array(carb_cost_l)*1e6,(np.array(price_index_l[country])-1)*100,lw=2,color=color)
    # carb_tax_eq = carb_cost_l[np.argmin(np.abs((np.array(price_index_l[country])-1)*100-infl[country]))]
    # ax1.scatter(carb_tax_eq*1e6,infl[country],lw=2,zorder=10)
if plot_inflation:
    ax1.scatter(np.array(carb_tax_eq_l)*1e6,np.array(inf_l),lw=4,zorder=10,c=sns.color_palette()[0:len(countries)],label='Inflation Feb22')
    ax1.set_yticks([y for y in np.linspace(0, 25, 11)])
    ax1.legend()
ax1.set_xticks([x for x in np.linspace(0,1000,11)])


if plot_inflation:
    leg = ax1.get_legend()
    leg.legendHandles[len(countries)].set_color('grey')
labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=20)  
ax1.tick_params(axis='y', labelsize = 20)

plt.tight_layout()

plt.show()


#%% Baseline 2018 - carbon cost = $100 plots #!!!!! needs to be ran to load data

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


#%% Sectoral price indices for a given country
country = 'CHE'
print('Plotting sectoral price indices for country '+country)

# Construct sectoral price index
p_hat_sol = sol_all[y].res.price_hat.to_numpy().reshape(C,S)

taxed_price = p_hat_sol*(1+carb_cost*sh['co2_intensity_np'])
price_agg_no_pow = np.einsum('it,itj->tj'
                                  ,taxed_price**(1-sigma)
                                  ,sh['share_cons_o_np']
                                  )
price_agg = np.divide(1,
                price_agg_no_pow ,
                out = np.ones_like(price_agg_no_pow),
                where = price_agg_no_pow!=0 ) ** (1/(sigma - 1))

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)


fig, ax = plt.subplots(figsize=(18,10))
color = sns.color_palette()[7]

data = ((price_agg[:,country_list.index(country)]-1)*100).tolist()
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
plt.suptitle('Sectoral consumer price change in '+country+' (%)',size = 30,y=0.96)

plt.tight_layout()

plt.show()

#%% Scatter plot of gross output change by country x sector for two carbon taxes

print('Plotting scatter plot of output changes for every country x sector according to produciton intensity')

fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

carb_cost_l = np.linspace(0.5e-4, 1.5e-4, 2)
color_list = [sns.color_palette()[1]
              # ,sns.color_palette("Paired")[7]
    , sns.color_palette("Paired")[1]
              ]

for c, carb_cost in enumerate(carb_cost_l):
    run_sc = runs.iloc[np.argmin(np.abs(runs['carb_cost'] - carb_cost))]
    res_sc = pd.read_csv(run_sc.path).set_index(['country', 'sector'])
    p_hat_sol_sc = res_sc.price_hat.to_numpy().reshape(C, S)
    E_hat_sol = res_sc.output_hat.to_numpy().reshape(C, S)
    q_hat_sol = E_hat_sol / p_hat_sol_sc
    q_hat_sol_percent = (q_hat_sol - 1) * 100

    ax.scatter(sh['co2_intensity_np'], q_hat_sol_percent, s=12, label=str(carb_cost * 1e6), color=color_list[c],
               zorder=1 - c)
    ax.set_ylabel('Production changes (%)',
                  fontsize=30
                  )
    ax.set_xscale('log')
    ax.set_ylim(-80, +40)
    # ax.set_xlim(1,3e3)
    ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)',
                  fontsize=30)

    ax.legend(['Tax = $50/ton of CO2', 'Tax = $150/ton of CO2'], fontsize=20, markerscale=5)
    ax.tick_params(axis='both', which='major', labelsize=20)

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.margins(x=0)
ax.hlines(0, xmin=sh['co2_intensity_np'].min(), xmax=sh['co2_intensity_np'].max(), colors='black', ls='--', lw=1)

# 50
sec = '20'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'RUS'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-250, 0),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '28'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'CHN'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-150, -100),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '35'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'NOR'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(20, 80),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '50'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'DEU'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(80, 15),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '01T02'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'BRA'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-250, -5),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '01T02'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'CHE'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(sh['co2_intensity_np'][country_index, sector_index], q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(100, -35),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

plt.show()

#%% Scatter plot of gross ouput change with kernel density by coarse industry

print('Plotting scatter plot of output changes for every country x sector according to produciton intensity with kernel density estimates for categories of sectors')

# p_hat_sol = sol_all[y].res.price_hat.to_numpy().reshape(C,S)
E_hat_sol = sol_all[y].res.output_hat.to_numpy().reshape(C,S)
q_hat_sol = E_hat_sol / p_hat_sol
q_hat_sol_percent = (q_hat_sol-1)*100

sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data = pd.DataFrame(data = q_hat_sol_percent.ravel(),
                    index = pd.MultiIndex.from_product([country_list,sector_map.index.get_level_values(level=0).to_list()],
                                                       names=['country','sector']),
                    columns=['value'])
data = data.reset_index().merge(sector_map.reset_index(),how='left',left_on='sector',right_on='ind_code').set_index(['country','sector']).drop('ind_code',axis=1)
data['co2_intensity'] = sh['co2_intensity_np'].ravel()
data['output'] = sh['output_np'].ravel()
data=data.sort_values('group_code')

sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)

group_labels_sorted = data.group_label.drop_duplicates().to_list()

data_no_z =data.copy()
data_no_z = data_no_z[data_no_z['co2_intensity'] != 0]
# data_no_z = data_no_z[data_no_z['co2_intensity'] < 1e4]
# data_no_z['co2_intensity'] = np.log(data_no_z['co2_intensity'])
data_no_z = data_no_z[['co2_intensity','value','group_label','group_code','output']]

data_no_z_1 = data_no_z[data_no_z['co2_intensity'] < 100].copy()
data_no_z_2 = data_no_z[data_no_z['co2_intensity'] >= 100].copy()

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)
# # sns.move_legend(plot, "lower left", bbox_to_anchor=(.55, .45), title='Species')

# plot.fig.get_axes()[0].legend(loc='lower left')
# # plt.legend(loc='lower left')

palette = [sns.color_palette('bright')[i] for i in [2,4,0,3,1,7]]
palette[0] = sns.color_palette()[2]
palette[1] = sns.color_palette("hls", 8)[-2]
for data_no_z_i in [data_no_z_1,data_no_z_2] :
# for data_no_z_i in [data_no_z] :
    plot2 = sns.kdeplot(data=data_no_z_i,
                x='co2_intensity',
                y="value",
                hue = 'group_label',
                fill = True,
                alpha = 0.25,
                log_scale=(True, False),
                # height=10,
                # ratio=5,
                # bw_adjust=0.5,
                weights = 'output',
                legend=False,
                levels = 2,
                palette = palette,
                common_norm = True,
                # shade=True,
                thresh = 0.2,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )
for data_no_z_i in [data_no_z_1,data_no_z_2] :
    for i,group in enumerate(data_no_z_i.group_code.drop_duplicates().to_list()):
        ax.scatter(data_no_z_i[data_no_z_i['group_code'] == group].co2_intensity,data_no_z_i[data_no_z_i['group_code'] == group].value,color=palette[i],s=10,zorder=1-i)

ax.set_ylabel('Production changes (%)',
                fontsize=30
                )
ax.set_xscale('log')
# ax.set_ylim(-100,+37.5)
ax.set_ylim(-80, +40)

# ax.set_xlim(0.5,20000)
ax.set_xlim(data_no_z.co2_intensity.min(),3e4)
ax.margins(x=0)
ax.tick_params(axis='both', which='major', labelsize=20)


ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)',
                fontsize=30)

handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax.legend(handles=handles,fontsize=20, loc = 'lower left')

ax.xaxis.set_major_formatter(ScalarFormatter())

ax.hlines(0,xmin=sh['co2_intensity_np'].min(),xmax=1e5,colors='black',ls='--',lw=1)


plt.show()

#%% Production reallocation

print('Computing production reallocation sector-wise')

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv', sep=';').set_index('ind_code')
sector_list = sol_all[y].output.index.get_level_values(1).drop_duplicates().to_list()
sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D' + sector].industry)

# Construct dataframe
output = sol_all[y].output

sector_dist = []
sector_change = []
sector_realloc = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
    sector_dist.append(pdist([output.xs(sector,level=1).value,output.xs(sector,level=1).new], metric = 'correlation')[0])
    temp = output.xs(sector,level=1).new-output.xs(sector,level=1).value
    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp>0].sum())
    sector_realloc_neg.append(temp[temp<0].sum())
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['output'] = output.groupby(level=1).sum().value.values
sector_dist_df['output_new'] = output.groupby(level=1).sum().new.values
sector_dist_df['distance'] = sector_dist
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

# sector_dist_df.sort_values('realloc_share_neg',ascending = False, inplace = True)

#%% Production reallocation, nominal differences

print('Plotting production reallocation in nominal differences')

sector_org = sector_dist_df[['industry', 'change', 'realloc','realloc_share_nom']].copy()
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

total_output_net_decrease = output.value.sum() - output.new.sum()
total_output = output.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated sector-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(27,-1),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)


ax.set_ylim(-1.25,0.55)

plt.show()

#%% Production reallocation, % changes

print('Plotting production reallocation in percentages')

sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg']].copy()
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

total_output_net_decrease = output.value.sum() - output.new.sum()
total_output = output.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated sector-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(27,-22.5),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)


ax.set_ylim(-32,12)

plt.show()

#%% Country specific reallocation of production

print('Computing production reallocation country-wise')

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()

country_dist = []
country_change = []
country_realloc = []
country_realloc_pos = []
country_realloc_neg = []
for country in country_list:
    country_dist.append(pdist([output.xs(country,level=0).value,output.xs(country,level=0).new], metric = 'correlation')[0])
    temp = output.xs(country,level=0).new-output.xs(country,level=0).value
    country_change.append(temp.sum())
    country_realloc_pos.append(temp[temp>0].sum())
    country_realloc_neg.append(temp[temp<0].sum())

country_dist_df = pd.DataFrame(index=country_list)
country_dist_df['output'] = output.groupby(level=0).sum().value.values
country_dist_df['output_new'] = output.groupby(level=0).sum().new.values
country_dist_df['distance'] = country_dist
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

total_output_net_decrease = output.value.sum() - output.new.sum()
total_output = output.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated country-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(41,-1.5),fontsize=25,zorder=10,backgroundcolor='w')

plt.show()

#%% Reallocation of production, percentage changes

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

total_output_net_decrease = output.value.sum() - output.new.sum()
total_output = output.value.sum()
total_output_decrease_percent = (total_output_net_decrease/total_output)*100

total_output_reallocated = np.abs(country_dist_df.realloc).sum()
total_output_reallocated_percent = (total_output_reallocated/total_output)*100

ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated country-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
             xy=(41,-16),fontsize=25,zorder=10,backgroundcolor='w')

plt.show()

#%% Within-country specific reallocation across sectors
country='USA'

print('Computing production reallocation in '+country)

output_country = sol_all[y].output.xs(country,drop_level=False)

sector_dist = []
sector_change = []
sector_realloc = []
sector_realloc_pos = []
sector_realloc_neg = []
for sector in sector_list:
    sector_dist.append(pdist([output_country.xs(sector,level=1).value,output_country.xs(sector,level=1).new], metric = 'correlation')[0])
    temp = output_country.xs(sector,level=1).new-output_country.xs(sector,level=1).value
    sector_change.append(temp.sum())
    sector_realloc_pos.append(temp[temp>0].sum())
    sector_realloc_neg.append(temp[temp<0].sum())
sector_map = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
sector_dist_df = sector_map.copy()
sector_dist_df['output_country'] = output_country.groupby(level=1).sum().value.values
sector_dist_df['output_country_new'] = output_country.groupby(level=1).sum().new.values
sector_dist_df['distance'] = sector_dist
sector_dist_df['realloc_pos'] = sector_realloc_pos
sector_dist_df['realloc_neg'] = sector_realloc_neg
sector_dist_df['change'] = sector_change

sector_dist_df['realloc_pos'] = np.abs(sector_dist_df['realloc_pos'])
sector_dist_df['realloc_neg'] = np.abs(sector_dist_df['realloc_neg'])
sector_dist_df['realloc'] = sector_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
sector_dist_df['realloc'] = sector_dist_df['realloc'] * np.sign(sector_dist_df['change'])
sector_dist_df['change_tot_nom'] = (sector_dist_df['change']+sector_dist_df['realloc'])
sector_dist_df['realloc_share_nom'] = (sector_dist_df['realloc']/sector_dist_df['change_tot_nom']) * np.sign(sector_dist_df['change'])

sector_dist_df['realloc_percent'] = (sector_dist_df['realloc']/sector_dist_df['output_country'])*100
sector_dist_df['change_percent'] = (sector_dist_df['change']/sector_dist_df['output_country'])*100
sector_dist_df['change_tot'] = (sector_dist_df['change_percent']+sector_dist_df['realloc_percent'])
sector_dist_df['realloc_share_neg'] = (sector_dist_df['realloc_percent']/sector_dist_df['change_tot']) * np.sign(sector_dist_df['change'])

# sector_dist_df.sort_values('realloc_share_neg',ascending = False, inplace = True)

#%% Production reallocation, nominal differences

print('Plotting production reallocation in nominal differences in '+country)

sector_use = sector_dist_df[['industry', 'change', 'realloc','realloc_share_nom']].copy()
sector_use.sort_values('change',ascending = True, inplace = True)
# sector_pos = sector_org[sector_org['realloc_share_nom']>0].copy()
# sector_pos.sort_values('change', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_nom']<= -0.15].copy()
# sector_neg1.sort_values('change',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_nom']> -0.15) & (sector_org['realloc_share_nom']<=0)].copy()
# sector_neg2.sort_values('change',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change/1e3
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of output in '+country,
            # color=colors
            )

# ax.bar(sector_use.industry
#             ,sector_use.realloc/1e6
#             ,bottom = sector_use.change/1e6
#             ,label='Reallocated output (%)',
#             # color=colors,
#             hatch="////")

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
ax.set_ylabel('Billion $', fontsize = 20)

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

# total_output_net_decrease = output.value.sum() - output.new.sum()
# total_output = output.value.sum()
# total_output_decrease_percent = (total_output_net_decrease/total_output)*100
#
# total_output_reallocated = np.abs(sector_dist_df.realloc).sum()
# total_output_reallocated_percent = (total_output_reallocated/total_output)*100

# ax.annotate('Overall, '+str(total_output_reallocated_percent.round(2))+'% of gross output\nwould be reallocated sector-wise for\na net reduction of output of '+str(total_output_decrease_percent.round(2))+'%',
#              xy=(27,-1),fontsize=25,zorder=10,backgroundcolor='w')

ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)


ax.set_ylim(-160, 85)

plt.show()

#%% Production reallocation, % changes

print('Plotting production reallocation in percentages in '+country)

sector_use = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg']].copy()
sector_use.sort_values('change_percent',ascending = True, inplace = True)
# sector_org = sector_dist_df[['industry', 'change_percent', 'realloc_percent','realloc_share_neg']].copy()
# sector_pos = sector_org[sector_org['realloc_share_neg']>0].copy()
# sector_pos.sort_values('change_percent', ascending = True, inplace = True)
# sector_neg1 = sector_org[sector_org['realloc_share_neg']<= -0.15].copy()
# sector_neg1.sort_values('change_percent',ascending = True, inplace = True)
# sector_neg2 = sector_org[(sector_org['realloc_share_neg']> -0.15) & (sector_org['realloc_share_neg']<=0)].copy()
# sector_neg2.sort_values('change_percent',ascending = True, inplace = True)
#
# sector_use = pd.concat([sector_neg2, sector_neg1, sector_pos], ignore_index=True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in sector_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(sector_use.industry
            ,sector_use.change_percent
            # ,bottom = sector_dist_df.realloc_neg
            ,label='Net change of output (%) in '+country,
            # color=colors
            )

# ax.bar(sector_use.industry
#             ,sector_use.realloc_percent
#             ,bottom = sector_use.change_percent
#             ,label='Reallocated output (%)',
#             # color=colors,
#             hatch="////")

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

ax.bar_label(ax.containers[0],
             labels=sector_use.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)


ax.set_ylim(sector_use.change_percent.min()-15,sector_use.change_percent.max()+26)

plt.show()

#%% Change in country's share of global output

print('Plotting change in countries share of global output of traded goods')

country_dist_df['share_o'] = country_dist_df.share_new_percent-country_dist_df.share_percent
country_dist_df.sort_values('share_o',ascending = True, inplace = True)

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

# palette = [sns.color_palette()[i] for i in [2,4,0,3,1,7]]
# colors = [palette[ind-1] for ind in country_dist_df.group_code]

# ax1=ax.twinx()

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.share_o
            # ,bottom = country_dist_df.realloc_neg
            ,label='Change in share of global production',
            # color=colors
            )

# ax.bar(country_dist_df.index.get_level_values(0)
#             ,country_dist_df.realloc_percent
#             ,bottom = country_dist_df.change_percent
#             ,label='Reallocated output',
#             # color=colors,
#             hatch="////")

ax.set_xticklabels(['']
                    , rotation=75
                    # , ha='right'
                    # , rotation_mode='anchor'
                    ,fontsize=19)
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percentage points',
              fontsize = 20)


leg = ax.legend(fontsize=20,loc='lower right')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-0.62,0.52)

ax.bar_label(ax.containers[0],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

plt.show()

#%% Load population data

print('Loading labor data')

labor = pd.read_csv('data/labor_force/labor.csv')
labor.set_index('country', inplace = True)
labor.sort_index(inplace = True)
labor_year = labor[year]

#%% Emissions reallocation

print('Computing emissions reallocation')

# Construct regional data 
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')

emissions = sol_all[y].co2_prod.groupby(level=0).sum()
emissions['change'] = -(emissions.new-emissions.value)
emissions = emissions.join(labor_year).rename(columns={year:'labor'})
emissions['per_capita'] = (emissions.value / emissions.labor) *1e6
emissions = emissions.join(country_map)

emissions.loc['TWN','Continent'] = 'Asia'
emissions.loc['ROW','Continent'] = 'Africa'
emissions.loc['AUS','Continent'] = 'Asia'
emissions.loc['NZL','Continent'] = 'Asia'
emissions.loc['CRI','Continent'] = 'South America'
emissions.loc['RUS','Continent'] = 'Asia'
emissions.loc['SAU','Continent'] = 'Africa'

#%% Emissions reduction relative to emissions per capita

print('Plotting emissions reduction relative to emissions per capita')

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }
colors = [continent_colors[emissions.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)
fig, ax = plt.subplots(figsize=(12,8),constrained_layout = False)

scatter = ax.scatter(emissions.per_capita,emissions.change,marker='x',lw=2,s=50,c = colors)


ax.set_xlabel('Emissions per capita (T.)', fontsize = 20)
ax.set_ylabel('Emission reduction (Mio. T)', fontsize = 20)

sns.kdeplot(data=emissions,
                x='per_capita',
                y="change",
                hue = 'Continent',
                fill = True,
                alpha = 0.15,
                log_scale=(False,True),
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                # weights = 'value',
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
handles = [mpatches.Patch(color=continent_colors[continent], label=continent) for continent in continent_colors.keys()]
# labels = continent_colors.keys()
ax.legend(handles = handles, loc="upper right", title="Continent")

# ax.set_yscale('log')
ax.set_ylim(1e-1,1e4)
ax.set_xlim(3,45)

texts = [plt.text(emissions.per_capita.loc[country], emissions.change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]
# adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )

plt.show()

#%% Emission reduction relative to emissions total

print('Plotting emissions reduction relative to emissions total')

palette = sns.color_palette()[0:5][::-1]
continent_colors = {
    'South America' : palette[0],
    'Asia' : palette[1],
    'Europe' : palette[2],
    'North America' : palette[3],
    'Africa' : palette[4],
                    }
colors = [continent_colors[emissions.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)
fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

scatter = ax.scatter(emissions.value,emissions.change,marker='x',lw=2,s=50,c = colors)

ax.set_xlabel('Total emissions country (Mio. T)', fontsize = 20)
ax.set_ylabel('Emission reduction (Mio. T)', fontsize = 20)

# sns.kdeplot(data=emissions,
#                 x='value',
#                 y="change",
#                 hue = 'Continent',
#                 fill = True,
#                 alpha = 0.15,
#                 log_scale=(True,True),
#                 # height=10,
#                 # ratio=5,
#                 # bw_adjust=0.7,
#                 # weights = 'value',
#                 # legend=False,
#                 levels = 2,
#                 palette = palette,
#                 # common_norm = False,
#                 shade=True,
#                 thresh = 0.1,
#                 # dropna=True,
#                 # fill = False,
#                 # alpha=0.6,
#                 # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
#                 ax = ax
#                 )
handles = [mpatches.Patch(color=continent_colors[continent], label=continent) for continent in continent_colors.keys()]
# labels = continent_colors.keys()
ax.legend(handles = handles, loc="lower right", title="Continent")

# test = np.polyfit(np.log(emissions.value),
#                   np.log(emissions.change),
#                   deg = 1,
#                   w=np.log(emissions.value)
#                   )
coeffs_fit = np.polyfit(np.log(emissions.value),
                  np.log(emissions.change),
                  deg = 1,
                  #w=emissions.labor
                  )


# ax.set_ylim(1e-4,1e1)
# ax.set_xlim(3e-3,20)

x_vals = np.arange(emissions.value.min()/10,emissions.value.max()*10)
y_vals = np.exp(coeffs_fit[1] + coeffs_fit[0] * np.log(x_vals))
ax.plot(x_vals, y_vals, '--',color='k')

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_ylim(1e-1,1e4)
ax.set_xlim(3,20e3)

texts = [plt.text(emissions.value.loc[country], emissions.change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]
# adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k', alpha=.5
                        )
        )

plt.show()


#%% Inequalities

print('Computing inequalities in terms of GDP change')

# Construct GDP per capita and welfare change
gdp = sol_all[y].va.groupby(level=0).sum()
gdp['price_index'] = price_index

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
colors = [continent_colors[gdp.loc[country,'Continent']] for country in country_list]
colors[country_list.index('RUS')] = (149/255, 143/255, 121/255)

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

ax.scatter(gdp.per_capita,gdp.utility_percent_change,marker='x',lw=2,s=50,c = colors)     # For kernel density
# ax.scatter(gdp.per_capita,gdp.utility_percent_change,marker='x',lw=2,s=50)

ax.set_xlabel('GDP per workforce (Thousands $)', fontsize = 20)
ax.set_ylabel('Welfare change (%)', fontsize = 20)

sns.kdeplot(data=gdp,
                x='per_capita',
                y="utility_percent_change",
                hue = 'Continent',
                fill = True,
                alpha = 0.25,
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                # weights = 'labor',
                # legend=False,
                levels = 2,
                palette = palette,
                # common_norm = False,
                shade=True,
                thresh = 0.12,
                # dropna=True,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )

# sns.move_legend(ax, "lower right")

ax.set_xlim(0,175)
# ax.set_ylim(-6.5,2.5)             # For kernel density
ax.set_ylim(-7,1.5)


texts = [plt.text(gdp.per_capita.loc[country], gdp.utility_percent_change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]     # For kernel density
# texts = [plt.text(gdp.per_capita.loc[country], gdp.utility_percent_change.loc[country], country,size=15) for i,country in enumerate(country_list)]

# adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))

plt.show()

#%% Correcting inequalities, the cost

print('Plotting adjustments needed for a fair tax')

labor = pd.read_csv('data/World bank/labor_force/labor.csv')
labor.set_index('country', inplace = True)
labor.sort_index(inplace = True)
labor_year = labor[year]

gdp = sol_all[y].va.groupby(level=0).sum()
gdp['price_index'] = price_index

gdp = gdp.join(labor_year).rename(columns={year:'labor'})
gdp['per_capita'] = (gdp.value / gdp.labor)

gdp['change']= (gdp.new/gdp.value)

gdp['change_real'] = ((gdp.new/gdp.value)/(gdp.price_index))

gdp['new_adjusted'] = gdp['change_real']*gdp['value']

# gdp_mean_change = gdp['new_adjusted'].sum()/gdp['value'].sum()
gdp_mean_change = gdp['new'].sum()/gdp['value'].sum()

gdp['new_if_average_change_adjusted']  = gdp['value'] * gdp_mean_change

# gdp['contribution'] = gdp['new_adjusted'] - gdp['new_if_average_change_adjusted']
gdp['contribution'] = gdp['new'] - gdp['new_if_average_change_adjusted']

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
# ax.set_ylim(-10000,8000)
# ax1.set_ylim(-10,8)

ax1.tick_params(axis = 'y', colors=sns.color_palette()[1] , labelsize = 20 )
ax1.set_ylabel('Contribution per worker (% of wage)', color = sns.color_palette()[1] , fontsize = 30)


plt.show()

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

ax.scatter(gdp.change,gdp.price_index)
ax.set_ylabel('CPI change' , fontsize = 20)
ax.set_xlabel('GDP change' , fontsize = 20)


annot_list = gdp.sort_values('wage_change_for_equality').index.to_list()[:5] \
    +gdp.sort_values('wage_change_for_equality',ascending=False).index.to_list()[:2]
# for cou in gdp.sort_values('wage_change_for_equality',ascending=False).index.to_list()[:5]
annot_list.append('RUS')
annot_list.append('BRA')
annot_list.append('ZAF')
annot_list.append('CHE')

for country in annot_list:
    ax.annotate(country,
                xy=(gdp.change.loc[country],gdp.price_index.loc[country]),
                xycoords='data',
                xytext=(0, 0),
                textcoords='offset points',
                va='center'
                )

plt.show()

#%% Computing trade data for connectivities

print('Computing trade data for connectivities')

traded = {}
tot = {}
# local = {}

for y in sol_all.keys():
    print(y)

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
    # cons_local = cons_local.loc[cons_local['row_country'] == cons_local['col_country']]  # = 0
    # cons_local = cons_local.set_index(['row_country', 'row_sector', 'col_country'])
    # cons_local = pd.concat({'cons': cons_local}, names=['col_sector']).reorder_levels([1, 2, 3, 0])

    # iot_local = iot.reset_index()
    # iot_local = iot_local.loc[iot_local['row_country'] == iot_local['col_country']]
    # iot_local = iot_local.set_index(['row_country', 'row_sector', 'col_country', 'col_sector'])

    traded[y] = pd.concat([iot_traded, cons_traded])
    # local[y] = pd.concat([iot_local, cons_local])
    tot[y] = pd.concat(
        [sol_all[y].iot, pd.concat({'cons': sol_all[y].cons}, names=['col_sector']).reorder_levels([1, 2, 3, 0])])

#%% Connectivity measure

print('Computing connectivities')

c_c = tot[y].groupby(level=[0,2]).sum()

c_co2_intensity = sol_all[y].co2_prod.groupby(level=0).sum().value / sol_all[y].output.groupby(level=0).sum().value

total_exchange = c_c.groupby(level=0).sum().rename_axis('country') + c_c.groupby(level=1).sum().rename_axis('country')

own_trade = c_c.reset_index()[c_c.reset_index().row_country == c_c.reset_index().col_country]
own_trade = own_trade.drop('col_country',axis=1).rename(columns={'row_country' : 'country'})
own_trade[['value' , 'new']] = own_trade[['value' , 'new']]*200
own_trade['change'] = own_trade.new - own_trade.value
own_trade['change_percent'] = (own_trade['change']/own_trade.value)*100
own_trade = own_trade.merge(total_exchange.reset_index(),suffixes = ['','_total_exchange'],on='country')
own_trade.set_index('country' , inplace=True)
own_trade['isolation'] = own_trade.value / own_trade.value_total_exchange
own_trade['isolation_new'] = own_trade.new / own_trade.new_total_exchange
own_trade['connectivity'] = 100- own_trade['isolation']
own_trade['connectivity_new'] = 100- own_trade['isolation_new']
own_trade['isolation_diff'] = own_trade['isolation_new'] - own_trade['isolation']
own_trade['connectivity_diff'] = own_trade['connectivity_new'] - own_trade['connectivity']

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

ax.scatter(own_trade['co2_intensity'] , own_trade['connectivity_diff'],lw=3,s=50,marker='x',c=colors)

sns.kdeplot(data=own_trade,
                x='co2_intensity',
                y="connectivity_diff",
                hue = 'Continent',
                fill = True,
                alpha = 0.2,
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                weights = 'labor',
                # legend=False,
                levels = 2,
                palette = palette,
                # common_norm = False,
                shade=True,
                thresh = 0.15,
                # dropna=True,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )

sns.move_legend(ax,'lower right')

coeffs_fit = np.polyfit(own_trade['co2_intensity'],
                  own_trade['connectivity_diff'],
                  deg = 1,
                  # w=own_trade.labor
                  )

x_lims = (0,900)
# y_lims = (-0.0025*200,0.0025*200)
ax.set_xlim(*x_lims)
y_lims = (-0.005*150,0.005*150)
ax.set_ylim(*y_lims)

x_vals = np.arange(0,x_lims[1])
y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')

ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')

ax.set_xlabel('CO2 intensity of production (Ton CO2 / $Mio.)',fontsize = 20)
ax.set_ylabel('Connectivity to the global trade, evolution prediction',fontsize = 20)

# plt.legend(loc='lower right')
# ax.legend(loc='lower right')

texts = [plt.text(own_trade['co2_intensity'].loc[country],  own_trade['connectivity_diff'].loc[country], country,size=15,color=colors[i]) for i,country in enumerate(country_list)]
adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

plt.show()

#%% Connectivity as a function of own connectivity

print('Plotting connectivity as a function of connectivity')

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

ax.scatter(own_trade['connectivity'] , own_trade['connectivity_diff'],lw=3,s=50,marker='x',c=colors)

sns.kdeplot(data=own_trade,
                x='connectivity',
                y="connectivity_diff",
                hue = 'Continent',
                fill = True,
                alpha = 0.2,
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                weights = 'labor',
                # legend=False,
                levels = 2,
                palette = palette,
                # common_norm = False,
                shade=True,
                thresh = 0.15,
                # dropna=True,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )

sns.move_legend(ax,'lower right')

coeffs_fit = np.polyfit(own_trade['connectivity'],
                  own_trade['connectivity_diff'],
                  deg = 1,
                    # w=own_trade.labor
                  )

x_lims = (0,55)
y_lims = (-0.005*150,0.005*150)
ax.set_xlim(*x_lims)
ax.set_ylim(*y_lims)

x_vals = np.arange(0,x_lims[1])
y_vals = coeffs_fit[1] + coeffs_fit[0] * x_vals
ax.plot(x_vals, y_vals, '-',lw=2,color='k',label='Regression line')

ax.hlines(y=0,xmin=x_lims[0],xmax=x_lims[1],ls='--',lw=1,color='k')

ax.set_xlabel('Connectivity to the global trade',fontsize = 20)
ax.set_ylabel('Connectivity to the global trade, evolution prediction',fontsize = 20)

# plt.legend(loc='lower right')
# ax.legend(loc='lower right')

texts = [plt.text(own_trade['connectivity'].loc[country],  own_trade['connectivity_diff'].loc[country], country,size=15,color=colors[i]) for i,country in enumerate(country_list)]
adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

plt.show()

#%% Historical plots - fixed carbon taxed or emissions target # !!!!!! load this and make choice

fixed_carb_tax = True #if True, will load historical data for a given carbon cost
carb_cost = 1e-4
adjust = True #if True, will adjust for dollar according to US inflation

emissions_target = False #if True, will load historical data for a given emissions target
reduction_target = 0.7 # emissions target in proportion of baseline emissions

if fixed_carb_tax:
    print('Loading historical data for fixed carbon tax')
    if carb_cost <= 1.5*1e-4:
        dir_num = 9
    else:
        dir_num = 8
    dollar_adjustment = pd.read_csv('data/dollar_adjustment.csv',sep=';',decimal=',').set_index('year')
    
    if adjust:
        sol_all_adjusted = {}
        for y in range(1995,2019):
            print(y)
            # print(dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost)
            sol_all_adjusted[y] = t.sol(y,dir_num,dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost)
        sol_all = sol_all_adjusted
    
    else:
        sol_all_unadjusted = {}
        for y in range(1995,2019):
            print(y)
            sol_all_unadjusted[y] = t.sol(y,dir_num,carb_cost)
        sol_all = sol_all_unadjusted
        
        
if emissions_target:
    print('Loading historical data for emissions reduction target')
    dir_num = 9
    sol_all_target = {}
    carb_tax = {}
    for y in range(1995,2019): 
        print(y)
        year = str(y)
        path = 'results/'+year+'_'+str(dir_num)
        runs = pd.read_csv(path+'/runs')
        run = runs.iloc[np.argmin(np.abs(runs.emissions-runs.iloc[0].emissions*reduction_target))]
        print('Carbon tax ='+str(run.carb_cost*1e6))    
        carb_tax[y] = run.carb_cost
        sol_all_target[y] = t.sol(y,dir_num,carb_tax[y])
    sol_all = sol_all_target

if (not emissions_target) & (not fixed_carb_tax):
    print('No data to load, make a choice on what to load')

#%% compute traded quantities

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
    cons_traded = pd.concat({'cons':cons_traded}, names=['col_sector']).reorder_levels([1,2,3,0])
    
    iot_traded = iot.reset_index()
    iot_traded = iot_traded.loc[iot_traded['row_country'] != iot_traded['col_country']]
    iot_traded = iot_traded.set_index(['row_country', 'row_sector', 'col_country','col_sector'])  
    
    # cons_local = cons.reset_index()
    # cons_local = cons_local.loc[cons_local['row_country'] == cons_local['col_country']]# = 0
    # cons_local = cons_local.set_index(['row_country', 'row_sector', 'col_country'])
    # cons_local = pd.concat({'cons':cons_local}, names=['col_sector']).reorder_levels([1,2,3,0])
    
    # iot_local = iot.reset_index()
    # iot_local = iot_local.loc[iot_local['row_country'] == iot_local['col_country']]
    # iot_local = iot_local.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
        
    
    traded[y] = pd.concat([iot_traded,cons_traded])
    # local[y] = pd.concat([iot_local,cons_local])
    tot[y] = pd.concat([sol_all[y].iot,pd.concat({'cons':sol_all[y].cons}, names=['col_sector']).reorder_levels([1,2,3,0])])

#%% compute distances for multiple quantities - function to compute distances

print('Computing distance indices for GSI comparison')

def compute_historic_distance(X):
    dist = []
    for i,y in enumerate(range(1995,2019)): 
        # print(y)
        dist.append(1 - pdist( [X[i].value , X[i].new] , metric = 'correlation'))
    return dist

years = [y for y in range(1995,2019)]
        
all_flows_dist = compute_historic_distance([tot[y] for y in years])
country_to_country_dist = compute_historic_distance([tot[y].groupby(level=[0,2]).sum() for y in years])
exports_dist = compute_historic_distance([tot[y].groupby(level=[0]).sum() for y in years])
imports_dist = compute_historic_distance([tot[y].groupby(level=[2]).sum() for y in years])
sectors = compute_historic_distance([tot[y].groupby(level=[1]).sum() for y in years])
c_s_c = compute_historic_distance([tot[y].groupby(level=[0,1,2]).sum() for y in years])
c_s = compute_historic_distance([tot[y].groupby(level=[0,1]).sum() for y in years])

all_flows_dist_traded = compute_historic_distance([traded[y] for y in years])
country_to_country_dist_traded = compute_historic_distance([traded[y].groupby(level=[0,2]).sum() for y in years])
exports_dist_traded = compute_historic_distance([traded[y].groupby(level=[0]).sum() for y in years])
imports_dist_traded = compute_historic_distance([traded[y].groupby(level=[2]).sum() for y in years])
sectors_traded = compute_historic_distance([traded[y].groupby(level=[1]).sum() for y in years])
c_s_c_traded = compute_historic_distance([traded[y].groupby(level=[0,1,2]).sum() for y in years])
c_s_traded = compute_historic_distance([traded[y].groupby(level=[0,1]).sum() for y in years])

# all_flows_dist_local = compute_historic_distance([local[y] for y in years])
# country_to_country_dist_local = compute_historic_distance([local[y].groupby(level=[0,2]).sum() for y in years])
# exports_dist_local = compute_historic_distance([local[y].groupby(level=[0]).sum() for y in years])
# imports_dist_local = compute_historic_distance([local[y].groupby(level=[2]).sum() for y in years])
# sectors_local = compute_historic_distance([local[y].groupby(level=[1]).sum() for y in years])
# c_s_c_local = compute_historic_distance([local[y].groupby(level=[0,1,2]).sum() for y in years])
# c_s_local = compute_historic_distance([local[y].groupby(level=[0,1]).sum() for y in years])    

gross_output_reduction_necessary = [-(tot[y].sum().new / tot[y].sum().value -1)*100 for y in years] 
share_traded_change = [(traded[y].sum().new/tot[y].sum().new - traded[y].sum().value/tot[y].sum().value)*100 for y in years]
welfare_change = [-((sol_all[y].utility.new*sol_all[y].cons.groupby(level=2).sum().new).sum()/sol_all[y].cons.sum().new-1)*100 for y in years]

#%% comparing GSI to welfare cost

print('Plotting different GSI choices and welfare cost of change')

fig, ax = plt.subplots(figsize=(12,8))

lw = 3
ax1 = ax.twinx()

traded_ls = '--'

ax.plot(years,imports_dist,label='Destination (imports) (1)',color = sns.color_palette()[5],lw=lw)
# ax.plot(years,imports_dist_traded,label='imports_dist_traded',color = sns.color_palette()[5],lw=lw,ls = traded_ls)

ax.plot(years,exports_dist,label='Origin (exports) (1)',color = sns.color_palette()[1],lw=lw)
# ax.plot(years,exports_dist_traded,label='exports_dist_traded',color = sns.color_palette()[1],lw=lw,ls = traded_ls)

ax.plot(years,country_to_country_dist,label='Origin x Destination (1)',color = sns.color_palette()[0],lw=lw)
# ax.plot(years,country_to_country_dist_traded,label='country_to_country_dist_traded',color = sns.color_palette()[0],lw=lw,ls = traded_ls)


ax.plot(years,sectors,label='Sectors (2)',color = sns.color_palette()[2],lw=lw)
# ax.plot(years,sectors_traded,label='sectors_dist_traded',color = sns.color_palette()[2],lw=lw,ls = traded_ls)


# ax.plot(years,c_s,label='c_s_dist',color = sns.color_palette()[4],lw=lw)
# ax.plot(years,c_s_traded,label='c_s_dist_traded',color = sns.color_palette()[4],lw=lw,ls = traded_ls)

ax.plot(years,c_s_c,label='Origin x Destination x Sectors (3)',color = sns.color_palette()[3],lw=lw)
# ax.plot(years,c_s_c_traded,label='c_s_c_dist_traded',color = sns.color_palette()[3],lw=lw,ls = traded_ls)

# ax.plot(years,all_flows_dist,label='all_flows_dist',color = sns.color_palette()[6],lw=lw)
# ax.plot(years,all_flows_dist_traded,label='all_flows_dist_traded',color = sns.color_palette()[6],lw=lw,ls = traded_ls)


# local_ls = ':'
# ax.plot(years,all_flows_dist_traded,label='all_flows_dist_traded')
# ax.plot(years,country_to_country_dist_local,label='country_to_country_dist_local',color = sns.color_palette()[0],lw=lw,ls = local_ls)
# ax.plot(years,exports_dist_local,label='exports_dist_local',color = sns.color_palette()[1],lw=lw,ls = local_ls)
# ax.plot(years,sectors_local,label='sectors_dist_local',color = sns.color_palette()[2],lw=lw,ls = local_ls)
# ax.plot(years,c_s_c_local,label='c_s_c_dist_local',color = sns.color_palette()[3],lw=lw,ls = local_ls)
# ax.plot(years,c_s_local,label='c_s_dist_local',color = sns.color_palette()[4],lw=lw,ls = local_ls)

# ax1.plot(years,gross_output_reduction_necessary,color = 'k', lw=2, ls= '--', label = 'Net output reduction')
ax1.plot(years,welfare_change,color = sns.color_palette()[0], lw=2, label = 'Welfare cost')
# ax1.plot(years,share_traded_change,color = 'k', lw=2)

ax.set_ylabel('GSI computed on different quantities (legend)',fontsize = 20,color = sns.color_palette()[3])
ax1.set_ylabel('Welfare cost of transition',fontsize = 20,color = sns.color_palette()[0])


# ax.legend(loc = (0.45,1.02),fontsize = 20, title = 'GSI computed on :')
# ax.legend(fontsize = 20, title = 'GSI computed on :')
# ax1.legend(fontsize = 20,loc = 'upper center')

ax1.grid(visible=False)

ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)

plt.show()

#%% Correlating GSI and welfare change 

print('Plotting correlation between different GSI choices and welfare cost of change')

# Correlation only exist if we take the reduction target point of view
fig, ax = plt.subplots(figsize=(12,8))

ax1 = ax.twinx()


ax.scatter([w for w in country_to_country_dist], welfare_change,color = sns.color_palette()[0],label='Country x Country index')
ax.scatter([w for w in sectors], welfare_change,color = sns.color_palette()[1],label='Sector index')
# ax.scatter([w for w in c_s], welfare_change,color = sns.color_palette()[2],label='Country x Sector of origin')
ax.scatter([w for w in c_s_c], welfare_change,label='GSI',color = sns.color_palette()[3])
# ax.scatter(imports_dist, welfare_change,label='imports_dist_traded',color = sns.color_palette()[5])

# ax1.scatter([1-w for w in country_to_country_dist_traded], gross_output_reduction_necessary,color = sns.color_palette()[0],marker='+', label='Country of origin\n(traded goods)')
# ax1.scatter([1-w for w in sectors], gross_output_reduction_necessary,color = sns.color_palette()[1],marker='+',label='Sector')
# ax1.scatter([1-w for w in c_s], gross_output_reduction_necessary,color = sns.color_palette()[2],marker='+',label='Country x Sector of origin')
# ax1.scatter([1-w for w in c_s_c], gross_output_reduction_necessary,label='c_s_c_dist',marker='+',color = sns.color_palette()[3])
# ax.scatter(imports_dist,  gross_output_reduction_necessary,label='imports_dist_traded',color = sns.color_palette()[5])

ax.legend()

ax1.grid(visible=False)
# ax1.legend(loc = (-0.5,0))

ax.set_xlabel('Index computed on different quantities (legend)')
ax.set_ylabel('Welfare cost (%)')
plt.title('Welfare cost of lowering emissions by  30%')

plt.show()

#%% just plotting GSI , comment country_country_dist plot line to have only GSI

print('Plotting GSI and geographical GSI')

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

lw = 4

# ax.plot(years,country_to_country_dist,label='Geographical GSI\n(Exporter x Importer)',lw=lw)
ax.plot(years,c_s_c,label='GSI',lw=lw)
# ax.set_yticks([])
ax.set_xticks(years)
ax.set_xticklabels(years
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=20)

# ax.legend(fontsize=20)
plt.title('Global Sustainability Index', fontsize = 20)
plt.savefig('../tax_model/eps_figures_for_ralph_pres/GSI.eps', format='eps')

plt.show()

#%% emissions saved every year by a fixed carbon tax

print('Plotting emissions reductions every year with fixed carbon tax if chosen')

if fixed_carb_tax:
    sol_all = sol_all_adjusted
    
    
    years = list(sorted(sol_all.keys()))
    
    emissions = [sol_all[y].co2_prod.sum() for y in years]
    emissions_saved = [(emissions[j].value - emissions[j].new)/1000 for j,y in enumerate(years)]
    emissions_saved_percent = [(emissions[j].value - emissions[j].new)*100/(emissions[j].value) for j,y in enumerate(years)]
    
    emissions_value = [emissions[j].value for j,y in enumerate(years)]
    emissions_new = [emissions[j].new for j,y in enumerate(years)]
    
    
    # corresp_tax = [dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost[y]*1e6 for y in years]
    # corresp_tax = [carb_cost[y]*1e6 for y in years]
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    # ax.plot(years,emissions_saved,lw=4)
    ax.plot(years,emissions_value,lw=4, label = 'Baseline emissions')
    ax.plot(years,emissions_new,lw=4, label = 'Counterfactual emissions')
    ax.set_xticks(years)
    
    ax.legend(loc = (0,0.25))
    
    ax.set_xticklabels(years
                       , rotation=45
                       , ha='right'
                       , rotation_mode='anchor'
                       ,fontsize=19)
    ax.set_title('Emissions reduction with a 100 dollars carbon tax adjusted'
                  ,fontsize=28
                  ,pad=15
                  )
    ax1 = ax.twinx()
    ax1.plot(years,emissions_saved_percent,lw=4,color = sns.color_palette()[3])
    
    
    ax.set_ylabel('Value (Gt)',fontsize=20)
    ax1.set_ylabel('Emissions reduction in percentage of baseline',fontsize=20,color = sns.color_palette()[3])
    
    ax1.grid(visible=False)
    
    ax.tick_params(axis='y', labelsize = 20)
    ax.margins(x=0.04)
    
    plt.tight_layout()
    
    plt.show()

#%% carbon_tax needed for a 30% emissions reduction

print('Plotting carbon tax needed every year with fixed emissions reduction target')

if emissions_target:
    sol_all = sol_all_target
    
    years = list(sol_all.keys())
    
    emissions = [sol_all[y].co2_prod.sum() for y in sol_all]
    emissions_saved = [(emissions[j].value - emissions[j].new)/1000 for j,y in enumerate(sol_all)]
    emissions_saved_percent = [(emissions[j].value - emissions[j].new)*100/(emissions[j].value) for j,y in enumerate(sol_all)]
    
    # corresp_tax = [dollar_adjustment.loc[y]['dollar_adjusted']*carb_cost[y]*1e6 for y in years]
    corresp_tax_adjusted = [carb_tax[y]*1e6/dollar_adjustment.loc[y].dollar_adjusted for y in years]
    corresp_tax = [carb_tax[y]*1e6 for y in years]
    
    fig, ax1 = plt.subplots(figsize=(12,8))
    
    # ax.plot(years,emissions_saved,lw=4)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years
                       , rotation=45
                       , ha='right'
                       , rotation_mode='anchor'
                       ,fontsize=19)
    ax1.set_title('To reduce emissions by 30%'
                  ,fontsize=28
                  ,pad=15
                  )
    # ax1 = ax.twinx()
    ax1.plot(years,corresp_tax,lw=4,color = sns.color_palette()[3])
    # ax1.plot(years,corresp_tax_adjusted,lw=4,color = sns.color_palette()[2], label = 'Dollar adjusted')
    
    # ax1.legend()
    
    # ax1.set_ylabel('Value (Gt)',fontsize=20,color = sns.color_palette()[0])
    ax1.set_ylabel('Carbon tax needed ($ per ton)',fontsize=20)
    
    # ax1.grid(visible=False)
    
    ax1.tick_params(axis='y', labelsize = 20)
    ax1.margins(x=0.04)
    
    plt.tight_layout()
    
    plt.show()

#%% evolution of change of countries output share

print('Plotting evolution of change of countries output share')

years = [y for y in range(1995,2019)]

country_country = [tot[y].groupby(level=[0]).sum() for y in years]

nbr_country = 11
biggest_countries = tot[y].groupby(level=[0]).sum().sort_values('value',ascending=False).index.get_level_values(0).to_list()[:nbr_country]

exports_country = {}

for country in biggest_countries:
    exports_country[country] = [100*((country_country[i].loc[country].new)/country_country[i].sum().new - (country_country[i].loc[country].value)/country_country[i].sum().value) for i,y in enumerate(years)]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

for country in biggest_countries:
    ax.plot(years,exports_country[country], label =country)
# ax.legend() 
# ax.set_yscale('log')
ax.set_xticks(years)
ax.set_xticklabels(years
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.margins(x=0.03)

ax.set_title('Predicted change in countries’ share of global output (p.p.)',fontsize = 20)

labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=15)   
plt.show()    

#%% evolution of countries output share

print('Plotting evolution of countries output share')

years = [y for y in range(1995,2019)]

country_country = [traded[y].groupby(level=[0]).sum() for y in years]

nbr_country = 11
biggest_countries = tot[y].groupby(level=[0]).sum().sort_values('value',ascending=False).index.get_level_values(0).to_list()[:nbr_country]

exports_share_country = {}

for country in biggest_countries:
    exports_share_country[country] = [100*(country_country[i].loc[country].value)/country_country[i].sum().value for i,y in enumerate(years)]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

for country in biggest_countries:
    ax.plot(years,exports_share_country[country], label =country)
# ax.legend() 
# ax.set_yscale('log')
ax.set_xticks(years)
ax.set_xticklabels(years
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.margins(x=0.03)

ax.set_title('Countries’ share of global output of traded goods (percentage)',fontsize = 20)

labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=15)   
plt.show()

#%% compute country specific time series

years = [y for y in range(1995,2019)]

def time_series_c(country):
    output = []
    share = []
    intensity = []
    share_change = []
    intensity_exports = []
    for y in range(1995,2019):
        print(y)
        temp = tot[y].value.groupby(level=0).sum()
        output.append(temp.loc[country])
        intensity.append(sol_all[y].co2_prod.value.groupby(level=0).sum().loc[country]*1e6 / sol_all[y].output.value.loc[country].to_numpy().sum())
        share.append(temp.loc[country] / tot[y].value.to_numpy().sum())    
        share_change.append( tot[y].new.groupby(level=[0]).sum().loc[country] / tot[y].new.to_numpy().sum() - temp.loc[country] / tot[y].value.to_numpy().sum() )
        intensity_exports.append(
            ((tot[y].groupby(level=[0,1]).sum().rename_axis(['country','sector']).value*sol_all[y].co2_intensity.value).groupby(level=0).sum()
            /temp).loc[country]
            )
    return share,intensity,output,share_change,intensity_exports


country = 'USA'

print('Computing time series of share,intensity,output,share_change,intensity_exports for country '+country)

share,intensity,output,share_change,intensity_exports = time_series_c(country)

# c_s_c = compute_historic_distance([tot[y].groupby(level=[0,1,2]).sum() for y in years])

#%% plot country specific time series

print('Plotting time series for country '+country)

c_c = compute_historic_distance([tot[y].groupby(level=[0,2]).sum() for y in years])

sns.set_style('whitegrid')

fig, ax = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)

color=sns.color_palette()[2]
ax[0,0].plot(years,c_c,lw=4,color=color)
# ax[0,0].set_title('GSI',fontsize=20)
ax[0,0].set_xlabel('')
ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,0].legend(['Country x country\nGSI World'])
ax[0,0].set_ylabel('GSI(2018) = 1',fontsize=20)

color=sns.color_palette()[1]
ax[1,0].plot(years,np.array(share_change)*100,lw=4,color=color)
# ax[0,1].set_title('Export share difference',fontsize=20)
# ax[1,0].set_ylim(-0.1,-0.4)
ax[0,1].set_xlabel('')
ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[1,0].legend(['Export share'])
ax[1,0].set_ylabel('Predicted difference (p.p.)',fontsize=20)


ax[1,1].plot(years,np.array(share)*100,lw=4)
ax[1,1].plot(years,(np.array(share)+np.array(share_change))*100,lw=4)
ax[1,1].set_ylabel('Export share (%)',fontsize=20)
# ax[1,0].set_title('Export share',fontsize=20)
ax[1,1].legend(['Data','Model'])
ax[1,1].yaxis.set_label_position("right")

color=sns.color_palette()[3]
ax[0,1].plot(years,intensity_exports,lw=4,color = color)
# ax[1,1].set_title('CO2 intensity of exports',fontsize=20)
ax[0,1].legend(['CO2 intensity of exports'])
ax[0,1].set_ylabel('Tons / Mio.$',fontsize=20)
ax[0,1].yaxis.set_label_position("right")

for i in [0,1]:
    for j in [0,1]:
        ax[i,j].tick_params(axis='y', direction='in')

# fig.suptitle(country,fontsize=30)

# plt.tight_layout()

plt.show()

#%% deeper look into intensities of a specific country

country = 'CHN'

print('Plotting scetor-wise intensities for country '+country)

years = [y for y in range(1995,2019)]

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/co2_intensity_prod_with_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']) for y in years],
    keys=years,names = ['year','country','sector']).reorder_levels([1,2,0]).sort_index()
output = pd.read_csv('data/yearly_CSV_agg_treated/datas2018/co2_intensity_prod_with_agri_ind_proc_fug_2018.csv').set_index(['country','sector'])

biggest_sectors = output.loc[country].sort_values('value',ascending=False).index.get_level_values(0).drop_duplicates()[:10]

intensities_country = intensities.loc[country]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

for sector in biggest_sectors:
    ax.plot(years,intensities_country.loc[sector].value,label=sector_map.loc['D'+sector].industry)

ax.set_yscale('log')
ax.set_title(country)
labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=15)   

plt.show()

#%% comparison of countries intensities 

print('Plotting comparison of countries intensities')

country_intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).groupby(level=0).sum() \
     / \
    pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).groupby(level=0).sum()
     for y in years],
    keys=years,names = ['year','country']).reorder_levels([1,0]).sort_index()

nbr_country = 11
biggest_countries = tot[y].groupby(level=[0]).sum().sort_values('value',ascending=False).index.get_level_values(0).to_list()[:nbr_country]

average_intensities = country_intensities.groupby(level=1).mean()
# average_intensities = pd.concat(
#     [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).sum() \
#       / \
#     pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).sum()
#       for y in years],
#     keys=years,names = ['year']).reorder_levels([1,0]).sort_index()

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

for country in biggest_countries:
    ax.plot(years,country_intensities.loc[country].value/average_intensities.value, label =country)
# ax.legend() 
# ax.set_yscale('log')
ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
ax.margins(x=0.03)
# ax.set_yscale('log')

ax.set_title('Countries average carbon intensity of production',fontsize = 20)

labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=15)   
plt.show()    

#%% comparison of countries intensities in one sector

sector = '01T02'

print('Plotting comparison of countries intensities in sector '+sector)

country_intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).xs(sector,level=1) \
     / \
    pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).xs(sector,level=1)
     for y in years],
    keys=years,names = ['year','country']).reorder_levels([1,0]).sort_index()

nbr_country = 10
biggest_countries = tot[y].xs(sector,level=1).groupby(level=0).sum().sort_values('value',ascending=False).index.get_level_values(0).drop_duplicates()[:nbr_country]

average_intensities = country_intensities.groupby(level=1).mean()


fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)


for country in biggest_countries:
    ax.plot(years,country_intensities.loc[country].value/average_intensities.value, label =country)
# ax.legend() 
# ax.set_yscale('log')
ax.set_xticks(years)
ax.set_xticklabels(years
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
ax.margins(x=0.03)
# ax.set_yscale('log')

ax.set_title('Countries average carbon intensity of production of sector '+sector_map.loc['D'+sector].industry,fontsize = 20)

labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=15)   
plt.show()  

#%% 4 subplots of carbon intensities

highlight = 'AUS'
nbr_country = 7
print('Plotting summary of carbon intensities for '+str(nbr_country)+' countries and highlighting '+highlight)

fig, ax = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)

lw_china = 3
lw_normal = 1.5

years = [y for y in range(1995,2019)]

# country_country = [traded[y].groupby(level=[0]).sum() for y in years]

biggest_countries = traded[y].groupby(level=[0]).sum().sort_values('value',ascending=False).index.get_level_values(0).to_list()[:nbr_country]
if highlight not in biggest_countries:
    biggest_countries.append(highlight)
biggest_countries[0],biggest_countries[3] = biggest_countries[3],biggest_countries[0]

exports_share_country = {}

for country in biggest_countries:
    exports_share_country[country] = [100*(country_country[i].loc[country].value)/country_country[i].sum().value for i,y in enumerate(years)]

for country in biggest_countries:
    if country == highlight:
        lw = lw_china
    else:
        lw = lw_normal
    ax[0,0].plot(years,exports_share_country[country], lw=lw, label =country)
# ax.legend() 
# ax.set_yscale('log')
# ax[0,0].set_xticks([])
ax[0,0].set_xticks(years)
ax[0,0].set_xticklabels([]
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=14)
ax[0,0].margins(x=0.03)

ax[0,0].set_title('Countries’ share of global output of traded goods (%)',fontsize = 14)

labelLines(ax[0,0].get_lines(),zorder=2.5,fontsize=14)   

country_intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).groupby(level=0).sum() \
     / \
    pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).groupby(level=0).sum()
     for y in years],
    keys=years,names = ['year','country']).reorder_levels([1,0]).sort_index()

average_intensities = country_intensities.groupby(level=1).mean()

for country in biggest_countries:
    if country == highlight:
        lw = lw_china
    else:
        lw = lw_normal
    ax[0,1].plot(years,country_intensities.loc[country].value/average_intensities.value, lw=lw, label =country)
# ax.legend() 
# ax.set_yscale('log')
ax[0,1].set_xticks(years)
ax[0,1].set_xticklabels([]
                   , rotation=45
                   , ha='right'
                   , rotation_mode='anchor'
                   ,fontsize=19)
ax[0,1].margins(x=0.03)
# ax.set_yscale('log')

ax[0,1].set_title('Countries average carbon intensity of production\ncomparative (1 is world average)',fontsize = 14)

labelLines(ax[0,1].get_lines(),zorder=2.5,fontsize=15)

sector = '01T02'

country_intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).xs(sector,level=1) \
     / \
    pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).xs(sector,level=1)
     for y in years],
    keys=years,names = ['year','country']).reorder_levels([1,0]).sort_index()

average_intensities = country_intensities.groupby(level=1).mean()

for country in biggest_countries:
    if country == highlight:
        lw = lw_china
    else:
        lw = lw_normal
    ax[1,0].plot(years,country_intensities.loc[country].value/average_intensities.value, lw=lw, label =country)

ax[1,0].set_xticks(years)
ax[1,0].set_xticklabels(years
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=12)
ax[1,0].tick_params(axis='x', which='major', pad=-10)
ax[1,0].margins(x=0.03)
# ax.set_yscale('log')

ax[1,0].set_title('Comparative carbon intensity of production\nsector '+sector_map.loc['D'+sector].industry+' (1 is world average)',fontsize = 14)

labelLines(ax[1,0].get_lines(),zorder=2.5,fontsize=15)   

sector = '24'

country_intensities = pd.concat(
    [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/prod_CO2_with_agri_agri_ind_proc_fug_'+str(y)+'.csv').set_index(['country','sector']).xs(sector,level=1) \
     / \
    pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(y)+'/output_'+str(y)+'.csv').set_index(['row_country','row_sector']).rename_axis(['country','sector']).xs(sector,level=1)
     for y in years],
    keys=years,names = ['year','country']).reorder_levels([1,0]).sort_index()

average_intensities = country_intensities.groupby(level=1).mean()

for country in biggest_countries:
    if country == highlight:
        lw = lw_china
    else:
        lw = lw_normal
    ax[1,1].plot(years,country_intensities.loc[country].value/average_intensities.value, lw=lw, label =country)

ax[1,1].set_xticks(years)
ax[1,1].set_xticklabels(years
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=12)
ax[1,1].tick_params(axis='x', which='major', pad=-10)
ax[1,1].margins(x=0.03)
# ax.set_yscale('log')

ax[1,1].set_title('Comparative carbon intensity of production\nsector '+sector_map.loc['D'+sector].industry+' (1 is world average)',fontsize = 14)

labelLines(ax[1,1].get_lines(),zorder=2.5,fontsize=15)   


plt.show()