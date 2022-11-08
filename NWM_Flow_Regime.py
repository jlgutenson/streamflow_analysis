#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import pandas as pd
import fsspec
import xarray as xr
import numpy as np
from dask import delayed, compute
from dask.distributed import Client, progress

import os


# In[2]:
if __name__ == '__main__':

    # load in EPA level III ecoregion data from
    # https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states
    path_to_ecoregion_shp = "/data/2023_NHD_Nevada/Shapefiles/us_eco_l3_state_boundaries_nad83.shp"
    ecoregions_gdf = gpd.read_file(path_to_ecoregion_shp)


    # In[3]:


    # load in NHDPlus v2.1 to read the NWM data in
    path_to_nhdplus_shp = "/data/2023_NHD_Nevada/Shapefiles/nhdflowlines_arizona.shp"
    flowlines_gdf = gpd.read_file(path_to_nhdplus_shp)


    # In[4]:


    # conduct spatial join between ecoregions and NHDPlus
    try:
        flowlines_gdf = flowlines_gdf.drop('index_left', axis=1)
    except:
        pass
    try:
        flowlines_gdf = flowlines_gdf.drop('index_right', axis=1)
    except:
        pass
    flowlines_gdf = gpd.sjoin(flowlines_gdf, ecoregions_gdf, how='left', op='intersects')


    # In[5]:


    # find all of the state of interest ecoregions
    state_of_interest = "Arizona"
    flowlines_state_of_interest_gdf = flowlines_gdf[flowlines_gdf['STATE_NAME']==state_of_interest]


    # In[6]:


    # built dictionary of the number of storms for each ecoregion from
    # https://www.researchgate.net/publication/268034350_Methods_for_development_of_planning-level_estimates_of_stormflow_at_unmonitored_sites_in_the_conterminous_United_States
    ecoregion_storms_dict = {
                             'Mojave Basin and Range' : 10,
                             'Colorado Plateaus' : 21,
                             'Arizona/New Mexico Plateau' : 20,
                             'Arizona/New Mexico Mountains' : 26,
                             'Chihuahuan Deserts' : 19,
                             'Madrean Archipelago' : 22,
                             'Sonoran Basin and Range' : 13
                            }

    flowlines_state_of_interest_gdf['storms_no'] = flowlines_state_of_interest_gdf['US_L3NAME'].map(ecoregion_storms_dict)


    # In[9]:


    # now lets query the NWM data to figure out how many days a year the streams in the state of interest flow
    river_ids_list = flowlines_state_of_interest_gdf['COMID'].to_list()
    # river_ids_list = river_ids_list[:10]

    def get_number_of_days(index, time_, dataset):
        try:

            timeseries = dataset['streamflow'].sel(time=time_,
                                                   feature_id=index).persist()
            data_ = timeseries.values

            qout_df = pd.DataFrame({"streamflow": data_}, index=time_)

            qout_df = qout_df[qout_df['streamflow']>0]

            # filter out the year 1979 because the NWM version 2.1 reanalysis starts in February, not January
            qout_df = qout_df[qout_df.index.year>1979]


            yearly_average_count = int(qout_df.groupby(qout_df.index.year).size().mean())
            print(yearly_average_count)
        except:
            yearly_average_count = 0

        return yearly_average_count

    # with Client() as client:
    client = Client()

    # output_streamflow_nc_path = "/data/2023_NHD_Nevada/Scripts/nwm_daily_mean_streamflow.nc"
    output_days_nc_path = "/data/2023_NHD_Nevada/Scripts/nwm_average_days_of_streamflow.nc"

    if os.path.exists(output_days_nc_path) is True:
        pass
    else:
        # access zarr dataset
        link = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'

        # create a MutableMapping from a store URL
        mapper = fsspec.get_mapper(link)

        # open the zarr using xarray
        ts_ds = xr.open_zarr(
          store=fsspec.get_mapper(link, anon=True),
          consolidated=True
        )

        # drop all the river reaches we don't need
        ts_subset_ds_0 = ts_ds.where(ts_ds.feature_id.isin(river_ids_list), drop=True)
        # average the hourly data into daily data
        ts_subset_ds_0 = ts_subset_ds_0.resample(time='D').mean()

        del(ts_ds)

        ts_subset_ds_1 = ts_subset_ds_0.copy()

        # count the number of days where streamflow is > 0
        ts_subset_ds_2 = ts_subset_ds_1.where(ts_subset_ds_1['streamflow']>0, drop=True)
        # print(ts_subset_ds_2)
        ts_subset_ds_2 = ts_subset_ds_2.streamflow.to_netcdf(output_days_nc_path)

        # # ouput xarray dataseries as a netcdf
        # ts_subset_ds_2 = ts_subset_ds_2.groupby("time.year").count().mean(dim="feature_id")
        # print(ts_subset_ds_2)
        #
        # ts_subset_ds_2 = ts_subset_ds_2.to_netcdf(output_days_nc_path)
        #
        # # ouput xarray dataseries as a netcdf
        # output_nc = ts_subset_ds_2.streamflow.to_netcdf(output_days_nc_path)

        print("finish outputting netcdf data...")

    # open the copy of NWM streamflow
    ts_ds = xr.open_dataset(output_days_nc_path)
    print("read netcdf data in...")

    try:
        del(ts_subset_ds_0)
        del(ts_subset_ds_1)
        del(ts_subset_ds_2)
    except:
        pass

    # pull time time array from the NWM netcdf data
    time_ = ts_ds["time"].values
    time_steps = len(time_)

    # create an empty dictionary to chart the average number of days per year with flow
    days_of_flow_dict = dict()

    number_of_days = 365
    # loop through the streams in your area of interest and add values to dictionary
    for i, rivid in enumerate(river_ids_list):
        print(f"working on {rivid}: {i+1} of {len(river_ids_list)}")
        yearly_average_count = get_number_of_days(rivid, time_, ts_ds)
        days_of_flow_dict[rivid] = yearly_average_count

    #combine data with NHDPlus and save as new output
    flowlines_state_of_interest_gdf['nwm_above_0'] = flowlines_state_of_interest_gdf['COMID'].map(days_of_flow_dict)
    flowlines_state_of_interest_gdf.loc[(flowlines_state_of_interest_gdf['nwm_above_0'] > flowlines_state_of_interest_gdf['storms_no']) & (flowlines_state_of_interest_gdf['nwm_above_0'] < number_of_days), 'regime'] = 'Intermittent'
    flowlines_state_of_interest_gdf.loc[(flowlines_state_of_interest_gdf['nwm_above_0'] <= flowlines_state_of_interest_gdf['storms_no']), 'regime'] = 'Ephemeral'
    flowlines_state_of_interest_gdf.loc[(flowlines_state_of_interest_gdf['nwm_above_0'] >= number_of_days), 'regime'] = 'Perennial'

    flowline_output_shp = "/data/2023_NHD_Nevada/Shapefiles/nhdflowlines_arizona_with_nwm_flow_count.shp"
    flowlines_state_of_interest_gdf.to_file(flowline_output_shp)
