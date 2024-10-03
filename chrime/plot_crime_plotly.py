import os,re
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from sklearn.linear_model import LinearRegression
import plotly.express as px

current_year = datetime.now().year
drop_years=[2001,2002]
year_xlims=[2002,current_year+1]

ordered_regions = [
    'Far North Side', None,
    'Northwest Side', 'North Side',
    'West Side', 'Central',
    'Southwest Side', 'South Side',
    'Far Southwest Side', 'Far Southeast Side'
]

select_crime_lists = {
    'weapons_and_homocide':['homicide','weapons'],
    'stealing':['larceny', 'burglary', 'robbery','vehicle_theft'],
    'assault_and_battery':['simple_battery','simple_assault','aggravated_battery', 'aggravated_assault','sexual_assault'],
    'society':['vandalism', 'misc', 'drug_abuse', 'fraud','sexual_abuse', 'disorderly_conduct']
}

used_ctypes = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT',
       'OTHER OFFENSE', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'ROBBERY',
       'WEAPONS VIOLATION','PUBLIC PEACE VIOLATION', 'SEX OFFENSE', 'CRIM SEXUAL ASSAULT',
       'INTERFERENCE WITH PUBLIC OFFICER','ARSON', 'HOMICIDE']

used_fbi_types = ['larceny', 'simple_battery', 'vandalism', 'misc', 'drug_abuse',
       'burglary', 'simple_assault', 'vehicle_theft', 'fraud', 'robbery',
       'aggravated_battery', 'aggravated_assault', 'weapons',
       'disorderly_conduct', 'sexual_assault', 'sexual_abuse',
       'family_offense', 'arson', 'homicide']

render_mode = 'SVG'

def melt_crime_df(
    crime_by_year_dfs,
    crime_type,
    place_type,
    drop_years=drop_years,
    population=None,
    community_areas=None,
    select_crimes=None,
    select_places=None,
    get_max_norm=True,
    pop_scale=1e3
):
    
    data_df = crime_by_year_dfs[(crime_type,place_type)].copy()
    
    if isinstance(select_crimes,str):
        select_crimes = select_crime_lists[select_crimes]
    elif select_crimes is None:
        select_crimes = used_fbi_types
    
    plot_data = (
        data_df
        .drop(drop_years)
        .unstack()
        .loc[select_crimes]
        .rename('n_crime')
    )
    if get_max_norm:
        df = (
            plot_data
            .reset_index()
            .set_index('Year')
            .groupby([crime_type,place_type],sort=False)
            .apply(lambda x: x/x.max())
            .rename({'n_crime':'n_crime_max-norm'},axis=1)
        )
        plot_data = pd.concat([plot_data,df],axis=1).reset_index()
    
    if population is not None:
        
        years = plot_data.Year.unique()
        years.sort()
        
        df_pop = get_pop_df(
            population,
            years,
            place_type,
            select_places
        )

        plot_data = plot_data.join(df_pop,on=[place_type,'Year'])
        plot_data['crime_rate'] = pop_scale*(plot_data['n_crime']/plot_data['Population'])
        
    plot_data = plot_data.dropna()
        
    if place_type == 'Community Area':
        plot_data[place_type] = community_areas.loc[plot_data[place_type],:]['Name'].values
        
    return plot_data

def get_pop_df(
    population,
    years,
    place_type,
    select_places
):

    p = interp1d(
    population.columns,
    population.loc[select_places,:],
    fill_value='extrapolate'
    )(years).T

    df_pop = pd.DataFrame(
        p,
        index=years,
        columns=select_places
    )
    df_pop.columns.name=place_type
    df_pop.index.name='Year'
    df_pop = df_pop.unstack().rename('Population')
    
    return df_pop

def plot_regional_crime(
    plot_data,
    crime_type,
    select_crimes=None
):
    
    fig = px.line(
        plot_data.round(decimals=2),
        x = 'Year',
        y = 'n_crime_max-norm',
        color = 'FBI_type',
        hover_data = {'Year':True,'FBI_type':True,'n_crime':True},
        labels={
            'FBI_type': 'Crime Category',
            'n_crime': 'Count',
            'n_crime_max-norm': 'Fraction of Max'
        },
        category_orders={'region':ordered_regions},
        facet_col='region',
        facet_col_wrap=2,
        height=900,
        facet_row_spacing=0.04,
        render_mode=render_mode
    )
    fig.update_traces(mode="markers+lines")
    # fig.update_layout(hovermode="x unified")

    for annotation in fig.layout.annotations:
        if annotation.text == 'region=None':
            annotation.update(text='') 

    fig.update_layout(
        legend=dict(
            x=0.675,  # x and y position of the legend (from 0 to 1)
            y=1,  # y position of the legend
        )
    )
    
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
    
    return fig

def plot_crime_by_type_and_place(
    plot_data,
    crime_type,
    place_type
):

    fig = px.line(
        plot_data.round(decimals=2),
        x = 'Year',
        y = 'crime_rate',
        color = place_type,
        hover_data = {'Year':True,'FBI_type':False,'n_crime':True,place_type:True},
        labels={
            'FBI_type': 'Crime Category',
            'n_crime': 'Count',
            'crime_rate': 'Per 1,000 capita'
        },
        category_orders={'FBI_type':used_fbi_types},
        facet_col='FBI_type',
        facet_col_wrap=4,
        height=900,
        width=1000,
        facet_row_spacing=0.06,
        facet_col_spacing=0.08,
        render_mode=render_mode
    )
    # fig.update_layout(hovermode="x unified")

    fig.update_yaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace('_',' ').title()))
    fig.update_xaxes(tickangle=45)

    return fig