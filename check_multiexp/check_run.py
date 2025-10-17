import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import glob

import yaml
import argparse
from pathlib import Path

###########################################################################################################

datadir = '../data/'
cart_out = './output/'

######################################################################################

def get_colors(exps):
    colorz = ['orange', 'steelblue', 'indianred', 'forestgreen', 'violet', 'maroon', 'teal', 'black', 'purple', 'olive', 'chocolate', 'dodgerblue', 'rosybrown', 'darkgoldenrod', 'lightseagreen', 'dimgrey', 'midnightblue']

    if len(exps) <= len(colorz):
        return colorz
    else:
        return colorz + get_spectral_colors(len(exps)-len(colorz))

def get_spectral_colors(n):
    """
    Extract n evenly spaced colors from the Spectral colormap.
    
    Parameters:
    n (int): Number of colors to extract
    
    Returns:
    list: List of RGB tuples
    """
    cmap = cm.get_cmap('nipy_spectral')
    colors = [cmap(i / (n - 1)) for i in range(n)]
    return colors


def add_diahsb_init_to_restart(rest_file, rest_file_new = None):
    """
    Adds missing fields to a restart (produced before 4.1.2) and creates a new restart (compatible with 4.1.2), filling missing variables with zeros.
    """

    rest_oce = xr.load_dataset(rest_file)

    # Get dimension sizes
    nt = len(rest_oce['time_counter'])
    ny = len(rest_oce['y'])
    nx = len(rest_oce['x'])
    nz = len(rest_oce['nav_lev'])

    # Add 0D variables (scalars)
    for var in 'v t s'.split():
        rest_oce[f'frc_{var}'] = xr.DataArray(0.0)

    # Add 2D variables (time_counter, y, x)
    for var in ['surf_ini', 'ssh_ini']:
        rest_oce[var] = xr.DataArray(
        np.zeros((nt, ny, nx)),
        dims=['time_counter', 'y', 'x'])

    # Add 3D variables (time_counter, nav_lev, y, x)
    for var in ['e3t_ini', 'tmask_ini', 'hc_loc_ini', 'sc_loc_ini']:
        rest_oce[var] = xr.DataArray(
        np.zeros((nt, nz, ny, nx)),
        dims=['time_counter', 'nav_lev', 'y', 'x'])
    
    if rest_file_new is None:
        rest_file_new = rest_file.replace('.nc', '_mod.nc')

    rest_oce.to_netcdf(rest_file_new)
    
    return


def read_temperature_copernicus(filepath = datadir + 'era5_daily_series_2t_global.csv'):
    """
    Read temperature CSV data into an xarray Dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
        
    Returns
    -------
    xr.Dataset
        Dataset with temperature variables indexed by date
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(
        filepath,
        comment='#',
        parse_dates=['date'],
        index_col='date'
    )
    
    # Drop the status column
    df = df.drop(columns=['status'])
    
    # Convert to xarray Dataset
    ds = xr.Dataset({
        '2t': ('time', df['2t'].values),
        'clim_91-20': ('time', df['clim_91-20'].values),
        'ano_91-20': ('time', df['ano_91-20'].values)
    }, coords={
        'time': np.array(df.index)
    })
    
    # Add attributes for metadata
    ds['2t'].attrs = {
        'long_name': 'Daily mean absolute temperature',
        'description': 'Based on hourly values from 00 to 23 UTC',
        'units': 'deg. C'
    }
    ds['clim_91-20'].attrs = {
        'long_name': 'Daily climatology for 1991-2020',
        'units': 'deg. C'
    }
    ds['ano_91-20'].attrs = {
        'long_name': 'Daily anomaly relative to 1991-2020 climatology',
        'units': 'deg. C'
    }
    
    #ds_clim = ds['clim_91-20'].groupby('time.dayofyear').mean()
    #ds.drop('clim_91-20')

    return ds#, ds_clim


def read_output(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce'.split()):
    """
    Reads outputs and computes global means.

    exps: list of experiment names to read
    user: list of users
    read_again: list of exps to read again (if run has proceeded)
    """

    if isinstance(user, str):
        user = len(exps)*[user]
    else:
        if len(user) != len(exps):
            raise ValueError(f"Length not corresponding: exps {len(exps)}, user {len(user)}")

    filz_exp = dict()
    filz_amoc = dict()
    filz_nemo = dict()

    for exp, us in zip(exps, user):
        cart = f'{cart_exp.format(us)}/{exp}/output/oifs/'
        filz = cart + f'{exp}_atm_cmip6_1m_*.nc'
        filz_exp[exp] = filz

    for exp, us in zip(exps, user):
        cart = f'{cart_exp.format(us)}/{exp}/output/nemo/'
        filz = cart + f'{exp}_oce_1m_diaptr3d_*.nc'
        filz_amoc[exp] = filz

        filz = cart + f'{exp}_oce_1m_T_*.nc'
        filz_nemo[exp] = filz

    atmmean_exp = dict()
    atmclim_exp = dict()
    oceclim = dict()
    amoc_mean_exp = dict()
    amoc_ts_exp = dict()


    for exp in exps:
        print(exp)
        coupled = False
        if len(glob.glob(filz_nemo[exp])) > 0:
            coupled = True

        if os.path.exists(cart_out + f'clim_tuning_{exp}.nc') and exp not in read_again:
            atmclim_exp[exp] = xr.open_dataset(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_exp[exp] = xr.open_dataset(cart_out + f'mean_tuning_{exp}.nc')
            if coupled:
                oceclim[exp] = xr.open_dataset(cart_out + f'oce_tuning_{exp}.nc')
        else:
            try:
                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})
            except OSError as err:
                print(err)
                print('Run still ongoing, removing last year')
                # Still running, remove last year
                fils = glob.glob(filz_exp[exp])
                fils.sort()
                filz_exp[exp] = fils[:-1]

                if coupled:
                    fils = glob.glob(filz_nemo[exp])
                    fils.sort()
                    filz_nemo[exp] = fils[:-1]

                    fils = glob.glob(filz_amoc[exp])
                    fils.sort()
                    filz_amoc[exp] = fils[:-1]

                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})

            ds = ds.rename({'time_counter': 'time'})
            ds = ds[atmvars].groupby('time.year').mean()

            atmclim_exp[exp] = ds.isel(year = slice(-20, None)).mean('year').compute()
            atmmean_exp[exp] = global_mean(ds, compute = True)
            atmclim_exp[exp].to_netcdf(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_exp[exp].to_netcdf(cart_out + f'mean_tuning_{exp}.nc')

            if coupled:
                ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                ds = ds.rename({'time_counter': 'time'})
                ds = ds[ocevars].groupby('time.year').mean()

                oceclim[exp] = ds.isel(year = slice(-20, None)).mean('year').compute()
                oceclim[exp].to_netcdf(cart_out + f'oce_tuning_{exp}.nc')

        if coupled:
            if os.path.exists(cart_out + f'amoc_ts_tuning_{exp}.nc') and exp not in read_again:
                amoc_ts_exp[exp] = xr.open_dataset(cart_out + f'amoc_ts_tuning_{exp}.nc')
                amoc_mean_exp[exp] = xr.open_dataset(cart_out + f'amoc_2d_tuning_{exp}.nc')
            else:
                ds = xr.open_mfdataset(filz_amoc[exp], use_cftime=True, chunks = {'time_counter': 240})
                amoc_ts = calc_amoc_ts(ds, plot = False)

                ds = ds.rename({'time_counter': 'time'})
                amoc = ds['msftyz'].groupby('time.year').mean()
                amoc = amoc.compute()
                amoc_mean = amoc.isel(year = slice(-30, None)).mean('year')
                amoc_mean = amoc_mean.squeeze()
                
                amoc_mean_exp[exp] = amoc_mean.compute()
                amoc_ts_exp[exp] = amoc_ts.compute()

                amoc_mean.to_netcdf(cart_out + f'amoc_2d_tuning_{exp}.nc')
                amoc_ts.to_netcdf(cart_out + f'amoc_ts_tuning_{exp}.nc')

    clim_all = dict()
    clim_all['atm_clim'] = atmclim_exp
    clim_all['atm_mean'] = atmmean_exp
    if coupled:
        clim_all['oce_clim'] = oceclim
        clim_all['amoc_mean'] = amoc_mean_exp
        clim_all['amoc_ts'] = amoc_ts_exp

    return clim_all


def plot_amoc_2d(amoc_mean, exp = None, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,8))

    if isinstance(amoc_mean, xr.Dataset):
        amoc_mean = amoc_mean['msftyz']

    amoc_mean.sel(basin = 2).plot.contourf(x = 'nav_lat', y = 'depthw', ylim = (3000, 0), levels = np.arange(-15, 15, 2), ax = ax)
    ax.set_title(exp)

    return ax


def calc_amoc_ts(data, ax = None, exp_name = 'exp', depth_min = 500., depth_max = 2000., lat_min = 38, lat_max = 50, ylim = (5, 20), plot = False, basin = 2):

    if plot and ax is None:
        fig, ax = plt.subplots()

    amoc = data.sel(
        depthw=slice(depth_min, depth_max), 
        basin=2
    )['msftyz']
    
    # Apply latitude constraint and compute
    amoc = amoc.where(
        (data['nav_lat'] > lat_min) & (data['nav_lat'] < lat_max)
    ).compute()
    
    # Resample to yearly means and find maximum
    amoc_yearly = amoc.resample(time_counter='YS').mean()
    amoc_max = amoc_yearly.max(dim=['depthw', 'y'])
    
    # Plot timeseries
    if plot:
        amoc_max.plot(ylim=ylim, label = exp_name, ax = ax)

    return amoc_max


def plot_amoc_ts(amoc_max, exp, ylim = (5, 20), ax = None, color = None, text_xshift = 10):
    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(amoc_max, xr.Dataset):
        amoc_max = amoc_max['msftyz']

    amoc_max = amoc_max.groupby('time_counter.year').mean()

    amoc_max.plot(ylim=ylim, label = exp, ax = ax, color = color)
    ax.text(amoc_max.year[-1]+text_xshift, amoc_max[-1], exp, fontsize=12, ha='right', color = color)

    return ax



def global_mean(ds, compute = True):
    try:
        all_lats = ds.lat.groupby('lat').mean()
        weights = np.cos(np.deg2rad(all_lats)).compute()
    except ValueError as coso:
        print(coso)
        print('Dask array, trying to use unique instead')
        all_lats = np.unique(ds.lat.values)
        weights = np.cos(np.deg2rad(all_lats))

    if 'time' in ds.coords:
        ds_mean = ds.groupby('time.year').mean().groupby('lat').mean().weighted(weights).mean('lat')
    elif 'year' in ds.coords:
        ds_mean = ds.groupby('lat').mean().weighted(weights).mean('lat')
    
    ds_mean = ds_mean['rsut rlut rsdt tas'.split()]
    ds_mean['toa_net'] = ds_mean.rsdt - ds_mean.rlut - ds_mean.rsut
    
    if compute:
        ds_mean = ds_mean.compute()

    return ds_mean



def plot_greg(atmmean_exp, exps, cart_out = cart_out, tas_clim = 287.29, net_toa_clim = 0.6, n_end = 20, imbalance = -0.9, ylim = None, colors = None):
    """
    gregory plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_betweenx(np.arange(0., 3.5, 0.1), tas_clim - 0.3, tas_clim + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_between(np.arange(286.5, 288, 0.1), net_toa_clim - imbalance - 0.3, net_toa_clim - imbalance + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        ax.plot(atmmean_exp[exp].tas, atmmean_exp[exp].toa_net, label = exp, lw = 0.2, color = col)
        #ax.scatter(atmmean_exp[exp].tas.sel(year = slice(1990, 2000)).mean(), atmmean_exp[exp].toa_net.sel(year = slice(1990, 2000)).mean(), s = 1000, color = 'red', marker = 'o')
        x, y = atmmean_exp[exp].tas.isel(year = slice(-n_end, None)).mean(), atmmean_exp[exp].toa_net.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', alpha = 0.5)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('net TOA (W/m$^2$)')
    plt.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.savefig(cart_out + f'check_tuning_{'-'.join(exps)}.pdf')
    plt.show()

    return fig



def plot_amoc_vs_gtas(clim_all, exps = None, cart_out = cart_out, tas_clim = 287.29, n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_betweenx(np.arange(4, 21, 0.1), tas_clim - 0.3, tas_clim + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_between(np.arange(286, 289, 0.1), 15, 20, color = 'grey', alpha = 0.2, edgecolor = None)

    if exps is None:
        exps = clim_all['atm_mean'].keys()

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        if isinstance(clim_all['amoc_ts'][exp], xr.DataArray):
            y = clim_all['amoc_ts'][exp]
        else:
            y = clim_all['amoc_ts'][exp]['msftyz']

        x = clim_all['atm_mean'][exp]['tas']
        y = y.groupby('time_counter.year').mean().squeeze()

        ax.plot(x, y, label = exp, lw = lw, color = col)
        
        x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('AMOC max (Sv)')

    if background_color is not None:
        ax.set_facecolor(background_color)

    if labels is None:
        plt.legend()
    else:
        if colors_legend is None:
            if len(colors) == len(labels):
                colors_legend = colors
            else:
                raise ValueError("specify colors for labels in legend")
        legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors_legend, labels)]

        ax.legend(handles=legend_elements)


    fig.savefig(cart_out + f'check_amoc_vs_gtas_{'-'.join(exps)}.pdf')
    plt.show()

    return fig


def create_ds_exp(exp_dict):
    x_ds = xr.concat(exp_dict.values(), dim=pd.Index(exp_dict.keys(), name='exp'))
    return x_ds

def plot_custom_greg(x_ds, y_ds, x_target, y_target, color_var = None, exps = None, cart_out = cart_out, n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None, cmap_name = 'viridis', xlabel = '', ylabel = '', cbar_label = ''):

    if isinstance(x_ds, dict):
        x_ds = xr.concat(x_ds.values(), dim=pd.Index(x_ds.keys(), name='exp'))
    if isinstance(y_ds, dict):
        y_ds = xr.concat(y_ds.values(), dim=pd.Index(y_ds.keys(), name='exp'))

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ext = (np.min(y_ds), np.max(y_ds))
    x_ext = (np.min(x_ds), np.max(x_ds))

    ax.fill_betweenx(np.arange(y_ext[0], y_ext[1], 0.1), x_target[0], x_target[1], color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_betweenx(np.arange(y_target[0], y_target[1], 0.1), x_ext[0], x_ext[1], color = 'grey', alpha = 0.2, edgecolor = None)

    if exps is None:
        exps = x_ds.exp.values

    if color_var is not None:
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=color_var.min(), vmax=color_var.max())

        colors = [cmap(norm(val)*0.8+0.1) for val in color_var]
    elif colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        x = x_ds.sel(exp = exp)
        y = y_ds.sel(exp = exp)

        ax.plot(x, y, label = exp, lw = lw, color = col)
        
        x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if background_color is not None:
        ax.set_facecolor(background_color)

    if labels is None:
        plt.legend()
    else:
        if colors_legend is None:
            if len(colors) == len(labels):
                colors_legend = colors
            else:
                raise ValueError("specify colors for labels in legend")
        legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors_legend, labels)]

        ax.legend(handles=legend_elements)

    # Add colorbar below the graph
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), 
                        ax=ax, orientation='vertical', pad=0.15)
    cbar.set_label(cbar_label)

    fig.savefig(cart_out + f'check_{x_ds.name}_vs_{y_ds.name}_{'-'.join(exps)}.pdf')
    plt.show()

    return


def plot_zonal_fluxes_vs_ceres(atm_clim, exps, plot_anomalies = True, weighted = False, datadir = datadir, cart_out = cart_out, colors = None, ylim = None):
    """
    plot_anomalies: plots anomalies wrt CERES
    weighted: weights for cosine of latitude
    """
    ceresmean = xr.open_dataset(datadir + 'ceres_clim_2001-2011.nc')

    atmclim = create_ds_exp(atm_clim)
    atmclim = atmclim.groupby('lat').mean()
    atmclim['toa_net'] = atmclim.rsdt - (atmclim.rsut + atmclim.rlut)

    if weighted:
        weights = np.cos(np.deg2rad(atmclim.lat)).compute()

    #####

    ceres_vars = ['toa_lw_all_mon', 'toa_sw_all_mon', 'toa_net_all_mon', 'solar_mon']
    okvars = ['rlut', 'rsut', 'toa_net', 'rsdt']

    figs = []
    for var, cvar in zip(okvars, ceres_vars):
        fig, ax = plt.subplots(figsize=(12, 8))
        y_ref = ceresmean.interp(lat = atmclim.lat)[cvar]
        
        if plot_anomalies: ax.axhline(0., color = 'lightgrey')
        if colors is None: colors = get_colors(exps)

        for exp, col in zip(exps, colors):
            y = atmclim.sel(exp = exp)[var]
            if plot_anomalies: y -= y_ref
            if weighted: y *= weights

            ax.plot(atmclim.lat, y, label = exp, color = col)
            ax.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
            
        if not plot_anomalies: 
            if weighted: y_ref *= weights
            ax.plot(atmclim.lat, y_ref, label = 'CERES', color = 'black')
            ax.text(100, y_ref.values[-1], 'CERES', fontsize=12, ha='right', color = 'black')

        ax.set_xlabel('lat')
        add = ''
        if weighted: add = ' (weighted with cosine)'
        if plot_anomalies:
            ax.set_ylabel(f'{var} bias wrt CERES 2001-2011 (W/m2)'+add)
        else:
            ax.set_ylabel(f'{var} vs CERES 2001-2011 (W/m2)'+add)

        plt.xlim(-90, 105)
        if ylim is not None: plt.ylim(ylim)
        #plt.legend()

        add = ''
        if not plot_anomalies: add = '_full'
        if weighted: add += '_weighted'

        fig.savefig(cart_out + f'check_radiation_vs_ceres_{'-'.join(exps)}{add}.pdf')
        figs.append(fig)

    return figs


def plot_map_ocean(oce_clim, exps, var, ref_exp = None, vmin = None, vmax = None, xlabel = None, ylabel = None):
    """
    oce_clim is clim_all['oce_clim'] produced by read_output
    exps: list of experiments
    var: var name to plot
    ref_exp: if specified, plot differences to ref_exp

    TO BE IMPROVED: regrid and plot with cartopy

    """
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))

    fig, axs = plt.subplots(nx, ny, figsize = (12, 12))
    for exp, ax in zip(exps, axs.flatten()):
        if ref_exp is not None:
            if exp == ref_exp:
                oce_clim[exp].tos.plot.pcolormesh(ax = ax)
                ax.set_title(exp)
            else:
                (oce_clim[exp]-oce_clim[ref_exp]).tos.plot.pcolormesh(vmin = vmin, vmax = vmax, ax = ax, cmap = 'RdBu_r')
                ax.set_title(f'{exp} - {ref_exp}')
        else:
            oce_clim[exp].tos.plot.pcolormesh(ax = ax, vmin = vmin, vmax = vmax)
            ax.set_title(exp)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    return fig


def plot_amoc_2d_all(amoc_mean, exps, cart_out = cart_out):
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))
    fig, axs = plt.subplots(nx, ny, figsize = (12, 12))

    for exp, ax in zip(exps, axs.flatten()):
        plot_amoc_2d(amoc_mean[exp], exp=exp, ax = ax)

    plt.tight_layout()
    fig.savefig(cart_out + f'check_amoc_2d_{'-'.join(exps)}.pdf')
    return fig


def plot_zonal_tas_vs_ref(atmclim, exps, ref_exp = None, cart_out = cart_out):
    # Missing tas reference
    atmclim = create_ds_exp(atmclim)
    atmclim = atmclim.groupby('lat').mean()

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ref = None
    if ref_exp is not None:
        y_ref = atmclim.sel(exp = ref_exp)['tas']

    colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = atmclim.sel(exp = exp)['tas']
        
        if y_ref is not None: y = y - y_ref

        plt.plot(atmclim.lat, y, label = exp, color = col)

        plt.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
        
    ax.axhline(0., color = 'grey')
    plt.xlim(-90, 105)
    #plt.legend()
    
    ax.set_xlabel('lat')
    if ref_exp is not None:
        ax.set_ylabel(f'zonal temp diff wrt {ref_exp} (K)')
    else:
        ax.set_ylabel('zonal temp (K)')

    fig.savefig(cart_out + f'check_zonal_tas_{'-'.join([exp for exp in exps if exp != ref_exp])}_vs_{ref_exp}.pdf')

    return fig


def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None):
    """
    Runs all multi-exps diagnostics.

    exps: list of experiments to consider
    cart_exp: base dir for experiments (defaults as $SCRATCH on hpc2020)
    user: to set experiment dir using cart_exp template. If a list, specifies a different user for every exp
    read_again: list of exps to read again. If set, overwrites existing clims for exp to update them (useful if sims are still running)
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    cart_out_nc = cart_out + '/exps_clim/'
    cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'

    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    if not os.path.exists(cart_out_figs): os.mkdir(cart_out_figs)

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc)

    fig_greg = plot_greg(clim_all['atm_mean'], exps, imbalance = imbalance, ylim = None, cart_out = cart_out_figs)
    fig_amoc_greg = plot_amoc_vs_gtas(clim_all, exps, lw = 0.25, cart_out = cart_out_figs)

    figs_rad = plot_zonal_fluxes_vs_ceres(clim_all['atm_clim'], exps = exps, cart_out = cart_out_figs)

    fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs)

    ###### CAN ADD NEW DIAGS HERE

    allfigs = [fig_greg, fig_amoc_greg] + figs_rad + [fig_tas]

    print(f'Done! Check results in {cart_out_figs}')

    return clim_all, allfigs


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path = None):
    if config_path is None:
        # Set up command line argument parser
        parser = argparse.ArgumentParser(description='Load configuration from YAML file')
        parser.add_argument('config', type=str, nargs='?', default='config.yml', help='Path to YAML configuration file')

        args = parser.parse_args()

        config_path = args.config
    
    # Load and parse configuration
    config = load_config(config_path)

    exps = config.get('exps', [])
    user = config.get('user', os.getenv('USER'))
    read_again = config.get('read_again', [])
    cart_exp = config.get('cart_exp', '/ec/res4/scratch/{}/ece4/')
    cart_out = config.get('cart_out')
    imbalance = config.get('imbalance')
    ref_exp = config.get('ref_exp')

    if user is None:
        user = os.getenv('USER')
    
    # Example: Print loaded configuration
    print(f"Experiments: {exps}")
    print(f"User: {user}")
    print(f"Read again: {read_again}")
    print(f"Cart exp: {cart_exp}")
    print(f"Cart out: {cart_out}")

    clim_all, figs = compare_multi_exps(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out, imbalance = imbalance, ref_exp = ref_exp)

    return clim_all, figs
    

# Main execution
if __name__ == '__main__':
    main()