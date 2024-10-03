import os
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms
from matplotlib.transforms import IdentityTransform

from scipy import ndimage
from scipy.interpolate import interp1d
from matplotlib import colormaps

current_year = datetime.now().year
drop_years=[2001,2002]
year_xlims=[2002,current_year+1]

major_ticks = np.arange(2000, year_xlims[1], 5)
minor_ticks = np.arange(year_xlims[0], year_xlims[1], 1)

used_ctypes = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT',
       'OTHER OFFENSE', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'ROBBERY',
       'WEAPONS VIOLATION','PUBLIC PEACE VIOLATION', 'SEX OFFENSE', 'CRIM SEXUAL ASSAULT',
       'INTERFERENCE WITH PUBLIC OFFICER','ARSON', 'HOMICIDE']

used_fbi_types = ['larceny', 'simple_battery', 'vandalism', 'misc', 'drug_abuse',
       'burglary', 'simple_assault', 'vehicle_theft', 'fraud', 'robbery',
       'aggravated_battery', 'aggravated_assault', 'weapons',
       'disorderly_conduct', 'sexual_assault', 'sexual_abuse',
       'family_offense', 'arson', 'homicide']

def init_crime_data(return_raw=False):
    base_dir = 'C:\\Users\\BatLab\\Documents\\chicago_crime'
    fname_str = 'Crimes_-_2001_to_Present_20240923.csv'

    raw_data_fname = os.path.join(base_dir,fname_str)

    comms_fname = os.path.join(base_dir,'community_areas.csv')
    demo_fname = os.path.join(base_dir,'ReferenceCCAProfiles20162020.csv')
    nibrs_fname = os.path.join(base_dir,'nibrs_codes.csv')
    iucr_fname = os.path.join(base_dir,'IUCR_Codes.csv')
    
    community_areas = pd.read_csv(comms_fname)
    community_areas = community_areas.set_index('CCAID')
    
    demographics = pd.read_csv(demo_fname)
    demographics = demographics.rename({'GEOID':'CCAID'},axis=1).set_index('CCAID').sort_index()
    
    nibrs_codes = pd.read_csv(nibrs_fname)    
    nibrs_codes.code = [f'{int(n):02}' if n.isnumeric() else n for n in nibrs_codes.code]
    nibrs_codes = nibrs_codes.set_index('code')

    iucr_codes = pd.read_csv(iucr_fname)
    iucr_codes = iucr_codes.set_index('IUCR')    
    
    population = demographics.iloc[:,1:4]
    population.columns = [2000,2010,2020]
    cca_regions = community_areas.Region.unique()
    for reg in cca_regions:
        cca_ids = community_areas.query('Region == @reg').index
        reg_pop = population.query('CCAID in @cca_ids').sum()
        population.loc[reg] = reg_pop
    
    if return_raw:
        raw_data = pd.read_csv(data_fname)
        raw_data['crimes_against'] = raw_data['FBI Code'].map(nibrs_codes.to_dict()['category'])
        raw_data['FBI_type'] = raw_data['FBI Code'].map(nibrs_codes.to_dict()['description'])
        raw_data['region'] = raw_data['Community Area'].map(community_areas.to_dict()['Region'])
    else:
        raw_data = None
        
    return community_areas, demographics, nibrs_codes, iucr_codes, population, raw_data

def get_commArea_demos(demographics,community_areas):
    df_demo_commArea = {}
    df_demo_commArea['medinc'] = demographics['MEDINC'].copy()
    df_demo_commArea['incpercap'] = demographics['INCPERCAP'].copy()
    df_demo_commArea['percemp'] = (demographics['UNEMP']/demographics['IN_LBFRC']).copy()
    df_demo_commArea['percvac'] = demographics['VACperc'].copy()

    df_demo_commArea = pd.DataFrame(df_demo_commArea)
    df_demo_commArea.index = df_demo_commArea.index.map(community_areas.Name)
    
    return df_demo_commArea

    
def get_crime_by_year_df(raw_data):
    
    data_pull_date = pd.to_datetime(raw_data.query('Year == @current_year').Date).max()
    
    crime_by_year_dfs = {}
    for crime_type,place_type in product(['Primary Type','crimes_against','FBI_type'],['Community Area','region']):
        crime_by_year_dfs[(crime_type,place_type)] = (
            raw_data
            .groupby([crime_type,place_type])
            .apply(lambda x: x['Year'].value_counts())
            .unstack([0,1])
            .fillna(value=0)
        )
        crime_by_year_dfs[(crime_type,place_type)].loc[data_pull_date.year]*=(365/data_pull_date.dayofyear)

def plot_regional_crime(
    crime_by_year_dfs,
    population,
    crime_type,
    select_crimes=None,
    drop_years=drop_years,
    xlim=year_xlims,
    legend_fontsize=8,
    show_max_text_box=False,
    plot_pop=False
):
    
    ordered_regions = [
        'Far North Side', None,
        'Northwest Side', 'North Side',
        'West Side', 'Central',
        'Southwest Side', 'South Side',
        'Far Southwest Side', 'Far Southeast Side'
    ]
    
    data_df = crime_by_year_dfs[(crime_type,'region')].copy()
    
    if (crime_type == 'crimes_against') and (select_crimes is None):
        select_crimes = np.unique(nibrs_codes['category'].values)
    elif (crime_type == 'FBI_type') and (select_crimes is None):
        select_crimes = used_fbi_types
    elif (crime_type == 'Primary Type') and (select_crimes is None):
        select_crimes = used_ctypes
        
    if len(select_crimes) < 5:
        figsize=(6,10)
    elif len(select_crimes) >=5:
        figsize=(6,13)
        
    # figsize=(14,10)
    
    fig,axes = plt.subplots(5,2,layout='tight',figsize=figsize,sharex=True,sharey=True)
    axes = axes.flatten()
    
    max_d = {}
    pop_axes = []
    for k,(ax, region) in enumerate(zip(axes,ordered_regions)):
        if region is None:
            ax.axis('off')
            ax.legend(objs,select_crimes+['population (1000\'s)'])
            pop_axes.append(None)
            continue
        
        d = data_df[select_crimes].xs(region,level='region',axis=1).drop(drop_years,errors='ignore')
        max_d[region] = d.max()
        plot_data = d/max_d[region]
        plot_data.plot(style='-x',ax=ax,legend=False,cmap='Accent')

        ax.set_title('\n'.join(region.split(' ')),loc='left')
        ax.set_ylabel('Fraction of max')
        ax.set_xlabel('Year')
        ax.set_xticks(major_ticks,labels=major_ticks,rotation=25)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.75)
        ax.set_xlim(xlim)
        
        if plot_pop:
            pop_ax = ax.twinx()
            p = (population.loc[region,:]/1000).astype(int)
            p.plot(style='--r',ax=pop_ax,alpha=0.5)
            if k%2 == 1:
                pop_ax.set_ylabel('Population', color='red')
            pop_ax.tick_params(axis='y', colors='red')

            pop_ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            
            objs = ax.get_lines() + pop_ax.get_lines()
            pop_axes.append(pop_ax)
        else:
            objs = ax.get_lines()
            pop_axes.append(None)
    
    
    if show_max_text_box:
        fig.canvas.draw()

        offset_x = 0.02

        for k,(ax, pop_ax, region) in enumerate(zip(axes, pop_axes, ordered_regions)):
            if region is None:
                continue
            label_fig_coords = {}
            for ax_type, used_ax in zip(['main','pop'],[ax,pop_ax]):
                label_box = used_ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
                label_fig_coords[ax_type] = fig.transFigure.inverted().transform(label_box)

            if k%2 == 0:
                text_x = label_fig_coords['main'][0][0] - offset_x
                ha='right'
            else:
                text_x = label_fig_coords['pop'][1][0] + offset_x
                ha='left'

            text_y = ax.get_position().y1

            legend_str = '\n'.join([f'{c}: max = {max_d[region][c]:.0f} ' for c in select_crimes])

            fig.text(
                text_x,text_y,legend_str,
                ha=ha,va='top',fontsize=legend_fontsize,
                bbox=dict(facecolor='white', alpha=0.5)
            )
        
        
    return fig

def plot_crime_by_type_and_place(
    crime_by_year_dfs,
    population,
    crime_type,
    place_type,
    community_areas=None,
    cca_regions=None,
    select_places=None,
    select_crimes=None,
    combine_regions=False,
    drop_years=drop_years,
    xlim=year_xlims
):
    
    data_df = crime_by_year_dfs[(crime_type,place_type)].copy()
    
    nrow,ncol = 5,4
    figsize=(11,11)
    
    if crime_type == 'crimes_against':
        if select_crimes is None:
            select_crimes = np.unique(nibrs_codes['category'].values)
    elif crime_type == 'FBI_type':
        if select_crimes is None:
            select_crimes = used_fbi_types
    elif crime_type == 'Primary Type':
        if select_crimes is None:
            select_crimes = used_ctypes
    
    n_crime = len(select_crimes)
    if (n_crime < 18) and (n_crime >= 16):
        nrow = 3
        ncol = 6
        figsize=(15,8)
    elif (n_crime <= 16) and (n_crime >= 12):
        nrow = 4
        figsize=(11,9)
    elif (n_crime < 12) and (n_crime >= 8):
        nrow = 3
        figsize=(11,6)
    elif (n_crime < 8) and (n_crime >= 5):
        nrow = 2
        figsize=(11,5)
    elif (n_crime < 5):
        nrow,ncol = 1,n_crime+1
        figsize=(11,3)

    if (place_type == 'Community Area') and (select_places is None):
        select_places = community_areas.index
    elif (place_type == 'region') and (select_places is None):
        select_places = cca_regions
    
    fig,axes = plt.subplots(nrow,ncol,sharex=True,layout='tight',figsize=figsize)
    axes = axes.flatten()
    _ = [ax.axis('off') for ax in axes]
    if len(select_crimes) == len(axes):
        legend_bbox = (1,1)
    else:
        legend_bbox = None

    for ax,ctype in zip(axes,select_crimes):
        d = data_df[ctype][select_places].drop(drop_years,errors='ignore')
        p = interp1d(population.columns,population.loc[select_places,:],fill_value='extrapolate')(d.index).T
        if combine_regions:
            plot_data = 1e3 * (d.sum(axis=1)/p.sum(axis=1))
        else:
            plot_data = 1e3 * (d/p)
            
        ax.axis('on')
        plot_data.plot(style='-x',ax=ax,legend=False,cmap='hsv',linewidth=1,markersize=5)
        objs = ax.get_lines()

        ax.set_title(ctype)
        ax.set_ylabel('Per 1000 capita')
        ax.set_xlabel('Year')
        ax.set_xticks(major_ticks,labels=major_ticks,rotation=25)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.75)
        ax.set_xlim(xlim)
    if place_type == 'Community Area':
        axes[-1].legend(objs,community_areas.loc[select_places,'Name'],bbox_to_anchor=legend_bbox)
    elif place_type == 'region':
        axes[-1].legend(objs,select_places,bbox_to_anchor=legend_bbox)
    
    return fig


def heatmap(
    d, 
    ax, 
    bins=(100,100),
    smoothing=1.3,
    cmap_str='Reds',
    use_colorbar=False,
    vrange=(None,None),
    scale = 1,
    density = False,
    cmap_thresh=0
):
    def getx(pt):
        return pt.coords[0][0]

    def gety(pt):
        return pt.coords[0][1]

    x = list(d.geometry.apply(getx))
    y = list(d.geometry.apply(gety))
    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins, density=density)
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    
    with np.errstate(divide='ignore'):
        logheatmap = np.log(heatmap)
    logheatmap[np.isneginf(logheatmap)] = 0
    logheatmap = ndimage.gaussian_filter(logheatmap, smoothing, mode='nearest')
    logheatmap *= scale
        
    logheatmap[logheatmap < cmap_thresh] = np.nan
    cmap = colormaps[cmap_str]
    cmap.set_bad('white',1.)
    vmin, vmax = vrange
    im = ax.imshow(logheatmap, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    if use_colorbar:
        plt.colorbar(im)
        
    return logheatmap, im
