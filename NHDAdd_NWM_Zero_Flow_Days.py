#!/usr/bin/env python
# coding: utf-8

# third-party libraries
import geopandas as gpd
import pandas as pd
import fsspec
import xarray as xr
import numpy as np
from dask import delayed, compute
from dask.distributed import Client, progress
import ulmo

# built-in libraries
import os
from datetime import datetime


def get_number_of_days(index, time_, dataset, begin_year, end_year, years_between):
    """
    Function to count the number of zero flow days that the NWM v2.1 estimates
    at each GageLoc location for the chosen beginning and ending year.
    Estimates a total number of zero flow days and a yearly average.
    """
    try:

        timeseries = dataset['streamflow'].sel(time=time_,
                                               feature_id=index).persist()
        data_ = timeseries.values

        qout_df = pd.DataFrame({"streamflow": data_}, index=time_)

        qout_df = qout_df[qout_df['streamflow']<=0]

        zero_flow_total = len(qout_df.index)
        zero_flow_average = zero_flow_total/years_between
    except:
        zero_flow_total = "nan"
        zero_flow_average = "nan"

    return zero_flow_total, zero_flow_average

# variables necessary to run the script
path_to_gage_shp = "/data/2023_NHD_Nevada/Shapefiles/GageLoc.shp" # available at https://www.sciencebase.gov/catalog/item/577445bee4b07657d1a991b6
start_date = datetime(2000, 1, 1) # changed from 1980
end_date = datetime(2020, 12, 31)
output_days_nc_path = "/data/2023_NHD_Nevada/Scripts/nwm_daily_streamflow_{0}_{1}.nc".format(start_date, end_date)
gage_filtered_shp = "/data/2023_NHD_Nevada/Shapefiles/GageLoc_with_zero_flow_{0}_{1}.shp".format(start_date, end_date)
gage_filtered_csv = "/data/2023_NHD_Nevada/Shapefiles/GageLoc_with_zero_flow_{0}_{1}.csv".format(start_date, end_date)

if __name__ == '__main__':

    # load in GageLoc data with NHD Gage Crosswalk
    gage_gdf = gpd.read_file(path_to_gage_shp, dtype={'SOURCE_FEA': 'str', 'FLComID': 'int'})

    # pull flow data from NWIS
    gage_id_list = gage_gdf['SOURCE_FEA'].to_list()
    years_between = ((end_date - start_date).days) / 365.25
    print(years_between)
    num_days = (end_date - start_date).days
    gage_id_success_list = []
    gage_zero_flow_days_dict = {}
    gage_average_yearly_zero_flow_days_dict = {}
    for gage_id in gage_id_list:
        # now lets pull the USGS NWIS data
        try:
            daily_data = ulmo.usgs.nwis.get_site_data(gage_id,
                                                      service='daily',
                                                      parameter_code='00060',
                                                      statistic_code='00003',
                                                      start=start_date,
                                                      end=end_date,
                                                      period=None,
                                                      modified_since=None,
                                                      input_file=None,
                                                      methods=None)
            daily_data_keys_list = list(daily_data.keys())
            dict_key = '00060:00003'
        except:
            pass
        #check to see if the dictionary is empty
        if not daily_data or dict_key not in daily_data_keys_list:
            pass
        else:
            gage_streamflow_df = pd.DataFrame(data=daily_data[dict_key]['values'])
            if len(gage_streamflow_df.index) < num_days or gage_streamflow_df['value'].isnull().values.any():
                pass
            else:
                gage_id_success_list.append(gage_id)
                gage_streamflow_df['value'] = pd.to_numeric(gage_streamflow_df['value'])
                gage_streamflow_df = gage_streamflow_df.loc[gage_streamflow_df['value']<=0]
                number_of_zero_flow_days = len(gage_streamflow_df.index)
                print(number_of_zero_flow_days)
                gage_zero_flow_days_dict[gage_id] = number_of_zero_flow_days
                average_number_of_zero_flow_days = number_of_zero_flow_days/years_between
                gage_average_yearly_zero_flow_days_dict[gage_id] = int(average_number_of_zero_flow_days)

    # use the `isin()` method to filter the dataframe
    gage_filtered_gdf = gage_gdf[gage_gdf['SOURCE_FEA'].isin(gage_id_success_list)]
    river_ids_list = gage_filtered_gdf['FLComID'].to_list()


    # save the NWM data as a local NetCDF or read the existing local NetCDF
    if os.path.exists(output_days_nc_path) is True:
        pass
    else:
        # with Client() as client:
        client = Client()

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
        ts_subset_ds_1 = ts_subset_ds_1.streamflow.to_netcdf(output_days_nc_path)

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

    # create an empty dictionary to chart the average number of days per year with flow
    nwm_zero_flow_days_dict = {}
    nwm_average_yearly_zero_flow_days_dict = {}

    # loop through the streams in your area of interest and add values to dictionary
    for i, rivid in enumerate(river_ids_list):
        print(f"working on {rivid}: {i+1} of {len(river_ids_list)}")
        zero_flow_total, zero_flow_average = get_number_of_days(rivid, time_, ts_ds, start_date.year, end_date.year, years_between)
        nwm_zero_flow_days_dict[rivid] = zero_flow_total
        nwm_average_yearly_zero_flow_days_dict[rivid] = zero_flow_average

    #combine data with gage data
    gage_filtered_gdf['usg_z'] = gage_filtered_gdf['SOURCE_FEA'].map(gage_zero_flow_days_dict)
    gage_filtered_gdf['usg_zavg'] = gage_filtered_gdf['SOURCE_FEA'].map(gage_average_yearly_zero_flow_days_dict)
    gage_filtered_gdf['nwm_z'] = gage_filtered_gdf['FLComID'].map(nwm_zero_flow_days_dict)
    gage_filtered_gdf['nwm_zavg'] = gage_filtered_gdf['FLComID'].map(nwm_average_yearly_zero_flow_days_dict)
    gage_filtered_gdf.to_file(gage_filtered_shp)

    # output the dataframe as a CSV file
    gage_filtered_gdf.to_csv(gage_filtered_csv, index=False)
    print("\n finished processing stuff...")
