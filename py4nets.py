#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:21:25 2022

@author: simonl
"""

import pandas as pd
import seaborn as sns
import py4cytoscape as p4c

#%% load solution

sol_all = {}
dir_num = 9
carb_cost = 1e-4
import treatment_funcs_agri_ind_fe as t

# for y in range(1995,2019):
for y in range(2018,2019):
    print(y)
    year = str(y)
    sol_all[y] = t.sol(y,dir_num,carb_cost)

sector_list = sol_all[y].iot.index.get_level_values(1).drop_duplicates().to_list()
S = len(sector_list)
country_list = sol_all[y].iot.index.get_level_values(0).drop_duplicates().to_list()
C = len(country_list)

#%% compute traded

traded = {}
tot = {}

for y in sol_all.keys():
    print(y)
    tot[y] = pd.concat([sol_all[y].iot,pd.concat({'cons':sol_all[y].cons}, names=['col_sector']).reorder_levels([1,2,3,0])])
    temp = tot[y].reset_index()
    traded[y] = temp[temp['row_country'] != temp['col_country']].set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    del temp

#%% functions

def pivot_df(df,columns):
    #makes columns with sectors
    df = df.pivot(index=columns,columns=['sector'])
    df.columns = df.columns.reorder_levels([1,0])
    df.sort_index(axis=1,inplace = True,ascending=False)
    df.columns = df.columns.to_flat_index()
    df.columns = [', '.join(c) for c in df.columns]
    df.reset_index(inplace=True)
    return df

def map_sectors(df,column):
    #replace ISIC codes with more explicit sector names
    sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';')
    sector_map['sector'] = df.sector.drop_duplicates()[:42].values
    sector_map.set_index('sector',inplace=True)
    df[column] = df[column].replace(sector_map['industry'])
    return df

def concatenate_total(df,columns):
    #compute the total of sectors, columns are the country columns
    #that will be used for the groupby
    total = df.groupby(columns,as_index = False).sum()
    total['sector'] = 'Total'
    df = pd.concat([df,total])
    return df

def compute_share_of_exports(traded,target_column,new_column_name):
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(traded.groupby(level=[1,2]).sum()[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_output(traded,tot,target_column,new_column_name):
    output = concatenate_total(
                map_sectors(tot.groupby(level=[0,1]).sum()
                            .rename_axis(['source','sector'])
                            .reset_index(),'sector'),['source']
                ).set_index(['source','sector'])
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = traded[target_column].div(output[target_column])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_imports(traded,target_column,new_column_name):
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(traded.groupby(level=[1,2]).sum()[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_demand(traded,tot,target_column,new_column_name):
    inputs = concatenate_total(
                map_sectors(tot.groupby(level=[1,2]).sum()
                            .rename_axis(['sector','target'])
                            .reset_index(),'sector'),['target']
                ).set_index(['sector','target'])
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(inputs[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def load_world_data():
    world = pd.read_csv('data/countries_codes_and_coordinates.csv')[['Alpha-3 code','Latitude (average)', 'Longitude (average)']]
    world.columns = ['country','latitude','longitude']
    for column in world.columns: 
        world[column] = world[column].str.replace('"','')
        world[column] = world[column].str.replace(' ','')
    world['country'] = world['country'].astype('string')
    world.drop_duplicates('country',inplace=True)
    world['latitude'] = -pd.to_numeric(world['latitude'])*10
    world['longitude'] = pd.to_numeric(world['longitude'])*10
    world.loc[world.index.max()+1] = ['ROW',0,0]
    return world

def compute_nodes_and_edges_baseline(traded, sol_all, y, world=None):
    #main function, computes the nodes and edges data from traded and sol_all
    #returns nodes and edges unpivoted = with a column "sectors" and the qty in columns
    #and nodes and edges pivoted = with columns for every sector X qty
    if world is None:
        world = load_world_data()
        
    # nodes   
    print('Computing nodes')
    nodes = traded[y].groupby(level=[0,1]).sum().reset_index().copy()
    nodes.columns = ['id','sector','output value','output new']
    nodes = pd.merge(nodes,traded[y].groupby(level=[1,2]).sum()
                         .rename_axis(['sector','id']).reset_index()
                     ,on = ['id','sector'])
    nodes.columns = ['id','sector','output value','output new','input value','input new']
    nodes = map_sectors(nodes,'sector')
    nodes = concatenate_total(nodes,['id'])
    
    nodes['output hat'] = nodes['output new']/nodes['output value']
    nodes['input hat'] = nodes['input new']/nodes['input value']
    
    output_share_traded = (concatenate_total(traded[y].groupby(level=[0,1]).sum()
                                              .reset_index()
                                              .rename(columns={'row_sector':'sector','row_country':'id'})
                                              ,['id']).set_index(['id','sector'])
                            / 
                            concatenate_total(tot[y].groupby(level=[0,1]).sum()
                                                                    .reset_index()
                                                                    .rename(columns={'row_sector':'sector','row_country':'id'})
                                                                    ,['id']).set_index(['id','sector'])).reset_index()
    output_share_traded.columns = ['id','sector','share output traded value','share output traded new']
    output_share_traded['share output traded hat'] = output_share_traded['share output traded new']/output_share_traded['share output traded value']
    
    input_share_traded = (concatenate_total(traded[y].groupby(level=[1,2]).sum()
                                              .reset_index()
                                              .rename(columns={'row_sector':'sector','col_country':'id'})
                                              ,['id']).set_index(['id','sector'])
                            / 
                            concatenate_total(tot[y].groupby(level=[1,2]).sum()
                                                                    .reset_index()
                                                                    .rename(columns={'row_sector':'sector','col_country':'id'})
                                                                    ,['id']).set_index(['id','sector'])).reset_index()
    input_share_traded.columns = ['id','sector','share input traded value','share input traded new']
    input_share_traded['share input traded hat'] = input_share_traded['share input traded new']/input_share_traded['share input traded value']
    
    
    output_share_traded = map_sectors(output_share_traded,'sector')
    input_share_traded = map_sectors(input_share_traded,'sector')
    
    nodes = pd.merge(nodes,output_share_traded)
    nodes = pd.merge(nodes,input_share_traded)
    
    nodes = nodes.fillna(1)
    nodes = nodes[['id', 'sector', 'output value', 'output hat', 'input value', 'input hat'
                   ,'share output traded hat', 'share input traded hat']]
    
    nodes_unpivoted = nodes
    nodes = pivot_df(nodes,['id'])
    nodes = nodes.merge(world,left_on='id',right_on='country').drop('country',axis=1)
    nodes['latitude'] = nodes['latitude']*1.2
    
    #edges    
    print('Computing edges')
    edges = traded[y].groupby(level=[0,1,2]).sum().reset_index().copy()
    edges.columns = ['source','sector','target', 'value', 'new']
    
    edges = map_sectors(edges,'sector')
    
    prices = sol_all[y].res.price_hat.reset_index()
    prices.columns = ['source','sector','price']
    prices = map_sectors(prices,'sector')
    edges = pd.merge(edges,prices,how='left',on=['source','sector'])
    edges['real new'] = edges['new'] / edges['price']
    
    edges = concatenate_total(edges,['source','target']).drop('price',axis=1)
    
    edges = compute_share_of_output(edges,tot[y],'value','share of output')
    edges = compute_share_of_output(edges,tot[y],'new','share of output new')
    
    edges = compute_share_of_exports(edges,'value','share of exports')
    edges = compute_share_of_exports(edges,'new','share of exports new')
    
    edges = compute_share_of_demand(edges,tot[y],'value','share of input')
    edges = compute_share_of_demand(edges,tot[y],'new','share of input new')
    
    edges = compute_share_of_imports(edges,'value','share of imports')
    edges = compute_share_of_imports(edges,'new','share of imports new')
    
    edges['hat'] = edges['new']/edges['value']
    edges['real hat'] = edges['real new']/edges['value']
    
    edges['share of output hat'] = edges['share of output new']/edges['share of output']
    edges['share of exports hat'] = edges['share of exports new']/edges['share of exports']
    edges['share of input hat'] = edges['share of input new']/edges['share of input']
    edges['share of imports hat'] = edges['share of imports new']/edges['share of imports']
    
    edges = edges.fillna(1)
    edges = edges[['source', 'sector', 'target', 'value','hat', 'real hat',
           'share of output hat', 'share of exports hat', 'share of input hat',
           'share of imports hat']]
    
    edges_unpivoted = edges
    edges = pivot_df(edges,['source','target'])
    
    return nodes_unpivoted, edges_unpivoted, nodes, edges
    
def create_baseline_network(nodes, edges, network_title):    
    #create network
    print('Creating network')
    network = p4c.create_network_from_data_frames(nodes = nodes, edges = edges, title=network_title)    
    
    return network
    
def create_network_for_specific_country(nodes_unpivoted, 
                                        edges_unpivoted, 
                                        country, 
                                        network_title, 
                                        input_or_output,
                                        world=None):
    nodes = nodes_unpivoted
    
    if world is None:
        world = load_world_data()
        
    if input_or_output == 'input':
        nodes = pd.merge(nodes,
                         edges_unpivoted[edges_unpivoted.target == country][['source','sector','share of input hat']], 
                         left_on = ['id','sector'],
                         right_on = ['source','sector'],
                         how='left'
                         ).drop(columns='source')
        nodes.loc[nodes['id'] == country, 'share of input hat'] = nodes.loc[nodes['id'] == country, 'share input traded hat']
        nodes_country = nodes[['id','sector','input value','share of input hat']]   
        edges_country = edges_unpivoted[edges_unpivoted.target == country]
        
    if input_or_output == 'output':
        nodes = pd.merge(nodes,
                         edges_unpivoted[edges_unpivoted.source == country][['target','sector','share of output hat']], 
                         left_on = ['id','sector'],
                         right_on = ['target','sector'],
                         how='left'
                         ).drop(columns='target')
        nodes.loc[nodes['id'] == country, 'share of output hat'] = nodes.loc[nodes['id'] == country, 'share output traded hat']
        
        nodes_country = nodes[['id','sector','output value','share of output hat']]   
        edges_country = edges_unpivoted[edges_unpivoted.source == country]
             
    nodes_country = pivot_df(nodes_country,['id'])
    nodes_country = nodes_country.merge(world,left_on='id',right_on='country').drop('country',axis=1)
    nodes_country['latitude'] = nodes_country['latitude']*1.2
        
    
    edges_country = pivot_df(edges_country,['source','target'])
    
    print('Creating network')
    network_country = p4c.create_network_from_data_frames(nodes = nodes_country, 
                                                          edges = edges_country, 
                                                          title = network_title)
        
    return nodes_country, edges_country, network_country

def c_map(df,sector,qty):
    data_points = [df[sector+', '+qty].quantile(i/10) for i in range(1,10)]
    color_points = sns.diverging_palette(15,120, s=65,l=50, n=9).as_hex()
    return data_points,color_points

class nodes_p:
    #class computes and outputs the actual values that will be mapped for colors and sizes for nodes
    def __init__(self,nodes,sector,node_size_max,
                 node_color_property_to_map,node_size_property_to_map):
        self.size_table_values = [0,nodes[sector+', '+node_size_property_to_map].max()]
        self.size_visual_values = [node_size_max/5,node_size_max]
        self.label_size_table_values = [0,nodes[sector+', '+node_size_property_to_map].max()]
        self.label_size_visual_values = [node_size_max/15,node_size_max/2.5]
        self.color_table_values = c_map(nodes,sector,node_color_property_to_map)[0]
        self.color_visual_values = c_map(nodes,sector,node_color_property_to_map)[1]
        
class edges_p:
    #class computes and outputs the actual values that will be mapped for colors and sizes for edges
    def __init__(self,edges,sector,edge_size_max,input_or_output,edge_color_property_to_map):
        self.size_table_values = [0,edges[sector+', value'].max()]
        self.size_visual_values = [0,edge_size_max]
        self.color_table_values = c_map(edges,sector,edge_color_property_to_map)[0]
        self.color_visual_values = c_map(edges,sector,edge_color_property_to_map)[1]
        self.transparency_table_values = [0,edges[sector+', value'].max()]
        self.transparency_visual_values = [100,255]
        if input_or_output == 'input':
            self.edge_source_arrow_shape = None
            self.edge_target_arrow_shape = 'DELTA'
        if input_or_output == 'output':
            self.edge_source_arrow_shape = 'CIRCLE'
            self.edge_target_arrow_shape = None
 
def make_title(sector, country, input_or_output):
    if country is None:
        if input_or_output == 'output':
            title = 'Exports'+' '+sector
        elif input_or_output == 'input':
            title = 'Imports'+' '+sector
    else:
        if input_or_output == 'output':
            title = 'Exports '+country+' '+sector
        elif input_or_output == 'input':
            title = 'Imports '+country+' '+sector
    return title        
 
#%% compute nodes and edges network

world = load_world_data()

nodes_unpivoted, edges_unpivoted, nodes_total, edges_total = \
    compute_nodes_and_edges_baseline(traded, sol_all, y,world)
    
#%% create and apply mapping
        
""" Sectors : ['Agriculture', 'Fishing', 'Mining, energy', 
'Mining, non-energy', 'Food products', 'Textiles', 
'Wood', 'Paper', 'Coke, petroleum', 'Chemicals', 
'Pharmaceuticals', 'Plastics', 'Non-metallic minerals', 
'Basic metals', 'Fabricated metals', 'Electronic', 
'Electrical equipment', 'Machinery', 'Transport equipments', 
'Manufacturing nec', 'Energy', 'Water supply', 'Construction', 
'Wholesale, retail', 'Land transport', 'Water transport', 'Air transport', 
'Warehousing', 'Post', 'Tourism', 'Media', 'Telecom', 'IT', 
'Finance, insurance', 'Real estate', 'R&D', 'Administration', 'Public sector', 
'Education', 'Health', 'Entertainment', 'Other service']
"""
y=2018
sector = 'Total'
country = None
print(sector)
print(country)
input_or_output = 'output'
bypass_existence_check = True
remake_network = True
remake_style = True

fit_view = True

# check if the network exists
title = make_title(sector, country, input_or_output)

if not bypass_existence_check: #check for already existing network if not bypassed
    if title not in p4c.get_network_list() or remake_network:
        try:
            p4c.delete_network(title)
        except:
            pass
        if country is None:
            network = create_baseline_network(nodes_total, edges_total, title)
        else:
            nodes_country, edges_country, network_country = \
                create_network_for_specific_country(nodes_unpivoted, 
                                                    edges_unpivoted, 
                                                    country, 
                                                    title, 
                                                    input_or_output, 
                                                    world)
    else:
        p4c.set_current_network(network=title)
        
if bypass_existence_check:
    if remake_network: #if check bypassed, can force the remake of the network
        if country is None:
            network = create_baseline_network(nodes_total, edges_total, title)
        else:
            nodes_country, edges_country, network_country = \
                create_network_for_specific_country(nodes_unpivoted, 
                                                    edges_unpivoted, 
                                                    country, 
                                                    title, 
                                                    input_or_output, 
                                                    world)
            
style_name = 'Style for '+sector+' '+title

if remake_style: #if remake of style is forced, delete style if it exists
    if style_name in p4c.get_visual_style_names():
        p4c.delete_visual_style(style_name)

if style_name not in p4c.get_visual_style_names(): #if style doesn't exist, build it
    
    print('Creating style')    

    node_size_max = 300
    edge_size_max = 60
   
    edge_color_property_to_map = 'share of '+input_or_output+' hat'
    node_size_property_to_map = input_or_output+' value'
    
    if country is None:
        node_color_property_to_map = 'share '+input_or_output+' traded hat'
        nodes = nodes_total
        edges = edges_total
    else:
        node_color_property_to_map = 'share of '+input_or_output+' hat'
        nodes = nodes_country
        edges = edges_country
        
    n = nodes_p(nodes,sector,node_size_max,node_color_property_to_map,node_size_property_to_map)
    e = edges_p(edges,sector,edge_size_max,input_or_output,edge_color_property_to_map)
        
    defaults = {'NODE_SHAPE': "circle",
                'NODE_FILL_COLOR': '#DCDCDC',
                'EDGE_TARGET_ARROW_SHAPE':e.edge_target_arrow_shape,
                'EDGE_SOURCE_ARROW_SHAPE':e.edge_source_arrow_shape,
                'EDGE_STACKING_DENSITY':0.7,
                # 'NODE_LABEL_POSITION':
                }
    n_labels = p4c.map_visual_property('node label', 'id', 'p')
    n_position_y = p4c.map_visual_property('NODE_Y_LOCATION', 'latitude', 'p')
    n_position_x = p4c.map_visual_property('NODE_X_LOCATION', 'longitude', 'p')
    n_position_z = p4c.map_visual_property('NODE_Z_LOCATION', sector+', '+node_size_property_to_map, 'p')
    n_size = p4c.map_visual_property('NODE_SIZE', sector+', '+node_size_property_to_map, 'c',
                                            table_column_values = n.size_table_values,
                                            visual_prop_values = n.size_visual_values )
    n_label_size = p4c.map_visual_property('NODE_LABEL_FONT_SIZE', sector+', '+node_size_property_to_map, 'c', 
                                           table_column_values = n.label_size_table_values,
                                           visual_prop_values = n.label_size_visual_values)

    n_color = p4c.map_visual_property('NODE_FILL_COLOR', sector+', '+node_color_property_to_map, 'c', 
                                            table_column_values = n.color_table_values,
                                            visual_prop_values = n.color_visual_values )

    e_stroke_color = p4c.map_visual_property('EDGE_STROKE_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                            table_column_values=e.color_table_values,
                                            visual_prop_values=e.color_visual_values )
    e_arrow_source_color = p4c.map_visual_property('EDGE_SOURCE_ARROW_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                            table_column_values=e.color_table_values,
                                            visual_prop_values=e.color_visual_values )
    e_arrow_target_color = p4c.map_visual_property('EDGE_TARGET_ARROW_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                            table_column_values=e.color_table_values,
                                            visual_prop_values=e.color_visual_values )
    e_width = p4c.map_visual_property('EDGE_WIDTH', sector+', value', 'c', 
                                           table_column_values = e.size_table_values,
                                           visual_prop_values = e.size_visual_values )
    e_arrow_target_width = p4c.map_visual_property('EDGE_TARGET_ARROW_SIZE', sector+', value', 'c', 
                                           table_column_values = e.size_table_values,
                                           visual_prop_values = e.size_visual_values )
    e_arrow_source_width = p4c.map_visual_property('EDGE_SOURCE_ARROW_SIZE', sector+', value', 'c', 
                                           table_column_values = e.size_table_values,
                                           visual_prop_values = e.size_visual_values)
    if country is None:
        e_transparency = p4c.map_visual_property('EDGE_TRANSPARENCY', sector+', value', 'c', 
                                               table_column_values = e.transparency_table_values,
                                               visual_prop_values = e.transparency_visual_values)
    else:
        e_transparency = p4c.map_visual_property('EDGE_TRANSPARENCY', sector+', value', 'c', 
                                               table_column_values = e.transparency_table_values,
                                               visual_prop_values = [255, 255])
    
    p4c.create_visual_style(style_name, defaults = defaults, mappings = [n_labels,
                                                                         n_position_y,
                                                                         n_position_x,
                                                                         n_position_z,
                                                                         n_size,
                                                                         n_label_size,
                                                                         n_color,
                                                                         e_width,
                                                                         e_stroke_color,
                                                                         e_arrow_target_color,
                                                                         e_arrow_source_color,
                                                                         e_arrow_target_width,
                                                                         e_arrow_source_width,
                                                                         e_transparency,
                                                                         ])
    
p4c.set_visual_style(style_name = style_name) #apply style

w = 3700
h = 2435*w/4378
if 'map' not in [anot['name'] for anot in p4c.get_annotation_list()]: #add map as annotation if not already there
    p4c.add_annotation_image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/World_map_blank_without_borders.svg/4378px-World_map_blank_without_borders.svg.png'
                              , x_pos=-1755.38, y_pos=-1028.95, opacity=100, brightness=0, contrast=None,
                              border_thickness=None, border_color=None, border_opacity=0, height=h, width=w,
                              name='map', canvas='background', z_order=None, network=None)

p4c.toggle_graphics_details()
# p4c.hide_all_panels()
# p4c.dock_panel('control')
# if not fit_view:
    # p4c.set_network_center_bypass(331,-142, bypass=True)
    # p4c.set_network_zoom_bypass(10, bypass=True)
# if fit_view:
p4c.fit_content()
if country is not None:
    p4c.set_node_property_bypass(country, 'diamond', 'NODE_SHAPE')
    p4c.add_annotation_text(text=title, x_pos=-1000, y_pos=800, font_size=100, font_family=None, font_style=None)
    p4c.set_node_property_bypass(country, 1e9, 'NODE_Z_LOCATION')
# p4c.set_network_property_bypass(10, 'NETWORK_SCALE_FACTOR')
# p4c.clear_network_zoom_bypass()
#%% choose color palette
# from numpy import arange
# x = arange(25).reshape(5, 5)
# cmap = sns.diverging_palette(10,120 , s=70,l=50, as_cmap=True)
# ax = sns.heatmap(x, cmap=cmap)


# temp = edges_unpivoted[(edges_unpivoted['source'] == 'ROW') & (edges_unpivoted['target'] == 'CHN')]
# temp.set_index('sector')['hat weighted']/edges_unpivoted['hat weighted'].mean()

