#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import fsspec
import xarray as xr
import numpy as np
from dask import delayed, compute
from dask.distributed import Client, progress
import ulmo

import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests


# function to parse the NWM netcdf data to retrieve one streamflow time series
def get_nwm_streamflow_timeseries(index, time_, dataset):

    timeseries = dataset['streamflow'].sel(time=time_,
                                           feature_id=index).persist()
    data_ = timeseries.values

    qout_df = pd.DataFrame({"streamflow": data_}, index=time_)

    return qout_df

# function for linear interpolation that we'll call later
def linear_interpolation(x, x0, y0, x1, y1):
    y = y0 + (x-x0)*((y1-y0)/(x1-x0))
    return y

# function for Nash-Sutcliffe that we'll call later
def nse(predictions, targets):
    nse = (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
    return nse

if __name__ == '__main__':

    # values we're going to collect for each USGS gage
    gage_ids = []
    comids = []
    lats = []
    lons = []
    streamflow_nses = []
    percentile_nses = []
    streamflow_rsqs = []
    percentile_rsqs = []

    # load in GageLoc data with NHDPlus-Gage Crosswalk
    path_to_gage_shp = "/data/2023_NHD_Nevada/Shapefiles/GageLoc.shp"
    gage_gdf = gpd.read_file(path_to_gage_shp, dtype={'SOURCE_FEA': 'str', 'FLComID': 'int'})
    river_ids_list = gage_gdf['FLComID'].to_list()

    # pull down NWM data using the GageLoc file
    output_days_nc_path = "/data/2023_NHD_Nevada/Scripts/nwm_daily_streamflow.nc"
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

    # pull time time array from the NWM netcdf data
    time_ = ts_ds["time"].values

    try:
        del(ts_subset_ds_0)
        del(ts_subset_ds_1)
        del(ts_subset_ds_2)
    except:
        pass

    # pull flow data from NWIS
    gage_id_list = gage_gdf['SOURCE_FEA'].to_list()
    start_date_test = datetime(2011, 1, 1)
    end_date_test = datetime(2020, 12, 31)
    test_dates = pd.date_range(start_date_test,end_date_test-timedelta(days=1),freq='d')
    gage_id_success_list = []
    gage_zero_flow_days_dict = {}
    gage_average_yearly_zero_flow_days_dict = {}

    # this is going to be a big loop through the GageLoc gages
    for gage_id in gage_id_list:
        print("trying {0}".format(gage_id))
        # try:
        # use the `isin()` method to filter the dataframe
        gage_filtered_gdf = gage_gdf[gage_gdf['SOURCE_FEA'].isin([gage_id])]
        rivid = gage_filtered_gdf['FLComID'].iloc[0]

        # pull the NWM time series
        nwm_df = get_nwm_streamflow_timeseries(rivid, time_, ts_ds)

        # filter the NWM data and create training nwm_percentiles
        nwm_filtered_df =  nwm_df.loc[:'2010-12-31']

        # generate percentile arrays for each day of the year, these will update each year when testing
        min_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).min()
        p05_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.05)
        p10_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.10)
        p20_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.20)
        p25_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.25)
        p50_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.50)
        p75_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.75)
        p80_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.80)
        p90_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.90)
        p95_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.95)
        max_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).max()

        # load our NWM test data
        nwm_test_df =  nwm_df.loc['2011-01-01':'2020-12-31']

        usgs_flows = []
        usgs_percentiles = []
        nwm_flows = []
        nwm_percentiles = []

        past_year = 2011

        for date in test_dates:
            print(date)
            # now lets pull the USGS NWIS data
            daily_data = ulmo.usgs.nwis.get_site_data(gage_id,
                                                      service='daily',
                                                      parameter_code='00060',
                                                      statistic_code='00003',
                                                      start=date.strftime('%Y-%m-%d'),
                                                      end=date.strftime('%Y-%m-%d'),
                                                      period=None,
                                                      modified_since=None,
                                                      input_file=None,
                                                      methods=None)
            daily_data_keys_list = list(daily_data.keys())
            dict_key = '00060:00003'
            #check to see if the dictionary is empty
            if not daily_data or dict_key not in daily_data_keys_list:
                daily_flow = "nan"
                streamflow_percentile = "nan"
                usgs_flows.append(daily_flow)
                usgs_percentiles.append(streamflow_percentile)
            else:
                value_list = daily_data[dict_key]['values']
                if len(value_list) == 0:
                    daily_flow = "nan"
                    streamflow_percentile = "nan"
                    usgs_flows.append(daily_flow)
                    usgs_percentiles.append(streamflow_percentile)
                else:
                    value_dict = value_list[0]
                    daily_flow = float(value_dict['value'])*0.0283168 # convert to cms
                    usgs_flows.append(daily_flow)
                    # pull the USGS gage location
                    lat = daily_data[dict_key]['site']['location']['latitude']
                    lon = daily_data[dict_key]['site']['location']['longitude']

                    # query the USGS stats REST Services
                    # had to use several temp files to receive and read USGS stats data
                    temp_file_1 = "usgs_statistics_temp_file_1.txt"
                    temp_file_2 = "usgs_statistics_temp_file_2.txt"
                    usgs_url = "https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites={0}&statReportType=daily&statTypeCd=all&parameterCd=00060".format(gage_id)
                    usgs_url_response = requests.get(usgs_url)
                    text_file = open(temp_file_1, "w")
                    n = text_file.write(usgs_url_response.text)
                    text_file.close()
                    text_file = open(temp_file_2, "w")
                    with open(temp_file_1) as f:
                        usgs_stats_lines = f.readlines()
                        for usgs_stat_line in usgs_stats_lines:
                            if usgs_stat_line.startswith("#") or usgs_stat_line.startswith("5s"):
                                pass
                            else:
                                 text_file.write(usgs_stat_line)
                    text_file.close()
                    usgs_stats_df = pd.read_csv(temp_file_2,delimiter='\t')
                    # filter the stats to just those for the day of interest
                    usgs_stats_df = usgs_stats_df[usgs_stats_df['month_nu']==date.month]
                    usgs_stats_df = usgs_stats_df[usgs_stats_df['day_nu']==date.day]
                    os.remove(temp_file_1)
                    os.remove(temp_file_2)
                    # print(usgs_stats_df)
                    min_va = usgs_stats_df['min_va'].iloc[0]* 0.0283168
                    p05_va = usgs_stats_df['p05_va'].iloc[0]* 0.0283168
                    p10_va = usgs_stats_df['p10_va'].iloc[0]* 0.0283168
                    p20_va = usgs_stats_df['p20_va'].iloc[0]* 0.0283168
                    p25_va = usgs_stats_df['p25_va'].iloc[0]* 0.0283168
                    p50_va = usgs_stats_df['p50_va'].iloc[0]* 0.0283168
                    p75_va = usgs_stats_df['p75_va'].iloc[0]* 0.0283168
                    p80_va = usgs_stats_df['p80_va'].iloc[0]* 0.0283168
                    p90_va = usgs_stats_df['p90_va'].iloc[0]* 0.0283168
                    p95_va = usgs_stats_df['p95_va'].iloc[0]* 0.0283168
                    max_va = usgs_stats_df['max_va'].iloc[0]* 0.0283168

                    # interpolating the daily observed USGS streamflow percentile here
                    if daily_flow <= min_va:
                        streamflow_percentile = 0.00
                    elif daily_flow >= min_va and daily_flow <= p05_va:
                        streamflow_percentile = linear_interpolation(daily_flow, min_va, 0, p05_va, 5)
                        # print("Between 0 and 5")
                    elif daily_flow > p05_va and daily_flow <= p10_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p05_va, 5, p10_va, 10)
                        # print(p)
                        # print("Between 5 and 10")
                    elif daily_flow > p10_va and daily_flow <= p20_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p10_va, 10, p20_va, 20)
                        # print(p)
                        # print("Between 10 and 20")
                    elif daily_flow > p20_va and daily_flow <= p25_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p20_va, 20, p25_va, 25)
                        # print(p)
                        # print("Between 20 and 25")
                    elif daily_flow > p25_va and daily_flow <= p50_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p25_va, 25, p50_va, 50)
                        # print(p)
                        # print("Between 25 and 50")
                    elif daily_flow > p50_va and daily_flow <= p75_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p50_va, 50, p75_va, 75)
                        # print(p)
                        # print("Between 50 and 75")
                    elif daily_flow > p75_va and daily_flow <= p80_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p75_va, 75, p80_va, 80)
                        # print(p)
                        # print("Between 75 and 80")
                    elif daily_flow > p80_va and daily_flow <= p90_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p80_va, 80, p90_va, 90)
                        # print(p)
                        # print("Between 80 and 90")
                    elif daily_flow > p90_va and daily_flow <= p95_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p90_va, 90, p95_va, 95)
                        # print(p)
                        # print("Between 90 and 95")
                    elif daily_flow > p95_va and daily_flow <= max_va:
                        streamflow_percentile = linear_interpolation(daily_flow, p95_va, 95, max_va, 100)
                        # print(p)
                        # print("Between 95 and 100")
                    elif daily_flow > max_va:
                        streamflow_percentile = 100.00
        #             print(streamflow_percentile)
                    usgs_percentiles.append(streamflow_percentile)

                    # determine when to update NWM perecentiles with additional data
                    current_year = date.year
                    if current_year > past_year:
                        print(current_year)
                        # recalculate nwm streamflow percentiles by adding this test day back to the percentiles
                        nwm_filtered_df =  nwm_df.loc[:date]
                        min_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).min()
                        p05_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.05)
                        p10_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.10)
                        p20_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.20)
                        p25_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.25)
                        p50_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.50)
                        p75_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.75)
                        p80_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.80)
                        p90_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.90)
                        p95_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).quantile(q=0.95)
                        max_va_nwm = nwm_filtered_df.groupby(pd.to_datetime(nwm_filtered_df.index).strftime('%m-%d')).max()
                        past_year = current_year

                    # now lets cycle through the NWM data like we did the USGS data
                    nwm_test_df_filtered = nwm_test_df.loc[date:date].values[0]
                    nwm_test_df_filtered = nwm_test_df_filtered[0]
                    nwm_flows.append(nwm_test_df_filtered)

                    # filter the daily flow duration curves by the date
                    month = str(date.month)
                    month = month.zfill(2)
                    day = str(date.day)
                    day = day.zfill(2)
                    min_va_nwm_filtered = min_va_nwm[min_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p05_va_nwm_filtered = p05_va_nwm[p05_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p10_va_nwm_filtered = p10_va_nwm[p10_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p20_va_nwm_filtered = p20_va_nwm[p20_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p25_va_nwm_filtered = p25_va_nwm[p25_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p50_va_nwm_filtered = p50_va_nwm[p50_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p75_va_nwm_filtered = p75_va_nwm[p75_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p80_va_nwm_filtered = p80_va_nwm[p80_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p90_va_nwm_filtered = p90_va_nwm[p90_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    p95_va_nwm_filtered = p95_va_nwm[p95_va_nwm.index == "{0}-{1}".format(month,day)].values[0]
                    max_va_nwm_filtered = max_va_nwm[max_va_nwm.index == "{0}-{1}".format(month,day)].values[0]

                    if nwm_test_df_filtered <= min_va_nwm_filtered:
                            streamflow_percentile = 0.00
                    elif nwm_test_df_filtered >= min_va_nwm_filtered and nwm_test_df_filtered <= p05_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, min_va_nwm_filtered, 0, p05_va_nwm_filtered, 5)
                        streamflow_percentile = streamflow_percentile
                        # print("Between 0 and 5")
                    elif nwm_test_df_filtered > p05_va_nwm_filtered and nwm_test_df_filtered <= p10_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p05_va_nwm_filtered, 5, p10_va_nwm_filtered, 10)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 5 and 10")
                    elif nwm_test_df_filtered > p10_va_nwm_filtered and nwm_test_df_filtered <= p20_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p10_va_nwm_filtered, 10, p20_va_nwm_filtered, 20)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 10 and 20")
                    elif nwm_test_df_filtered > p20_va_nwm_filtered and nwm_test_df_filtered <= p25_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p20_va_nwm_filtered, 20, p25_va_nwm_filtered, 25)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 20 and 25")
                    elif nwm_test_df_filtered > p25_va_nwm_filtered and nwm_test_df_filtered <= p50_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p25_va_nwm_filtered, 25, p50_va_nwm_filtered, 50)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 25 and 50")
                    elif nwm_test_df_filtered > p50_va_nwm_filtered and nwm_test_df_filtered <= p75_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p50_va_nwm_filtered, 50, p75_va_nwm_filtered, 75)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 50 and 75")
                    elif nwm_test_df_filtered > p75_va_nwm_filtered and nwm_test_df_filtered <= p80_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p75_va_nwm_filtered, 75, p80_va_nwm_filtered, 80)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 75 and 80")
                    elif nwm_test_df_filtered > p80_va_nwm_filtered and nwm_test_df_filtered <= p90_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p80_va_nwm_filtered, 80, p90_va_nwm_filtered, 90)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 80 and 90")
                    elif nwm_test_df_filtered > p90_va_nwm_filtered and nwm_test_df_filtered <= p95_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p90_va_nwm_filtered, 90, p95_va_nwm_filtered, 95)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 90 and 95")
                    elif nwm_test_df_filtered > p95_va_nwm_filtered and nwm_test_df_filtered <= max_va_nwm_filtered:
                        streamflow_percentile = linear_interpolation(nwm_test_df_filtered, p95_va_nwm_filtered, 95, max_va_nwm_filtered, 100)
                        streamflow_percentile = streamflow_percentile
                        # print(p)
                        # print("Between 95 and 100")
                    elif nwm_test_df_filtered > max_va_nwm_filtered:
                        streamflow_percentile = 100.00
                    nwm_percentiles.append(streamflow_percentile)

        # generate plot comparing the percentiles
        test_dates = nwm_test_df.index.tolist()

        timeseries_data = {
            'date': test_dates,

            'nwm_flow': nwm_flows,

            'nwm_percentiles' : nwm_percentiles,

            'usgs_flows' : usgs_flows,

            'usgs_percentiles' : usgs_percentiles
        }

        # Create dataframe
        dataframe = pd.DataFrame(timeseries_data,columns=['date','nwm_flow','nwm_percentiles','usgs_flows','usgs_percentiles'])

        # Setting the Date as index
        dataframe = dataframe.set_index("date")

        # Calculate NSE
        try:
            nwm_nse = nse(dataframe['nwm_flow'], dataframe['usgs_flows'])
            streamflow_nses.append(nwm_nse)
        except:
            nwm_nse = "nan"
            streamflow_nses.append(nwm_nse)
        try:
            nwm_nse = nse(dataframe['nwm_percentiles'], dataframe['usgs_percentiles'])
            percentile_nses.append(nwm_nse)
        except:
            nwm_nsw = "nan"
            percentile_nses.append(nwm_nse)

        # Calculate R-Squared
        try:
            #initiate linear regression model
            model = LinearRegression()
            X, y = dataframe[['nwm_flow']], dataframe['usgs_flows']
            #fit regression model
            model.fit(X, y)
            #calculate R-squared of regression model
            r_squared = model.score(X, y)
            streamflow_rsqs.append(r_squared)
        except:
            r_squared = "nan"
            streamflow_rsqs.append(r_squared)

        try:
            #initiate linear regression model
            model = LinearRegression()
            X, y = dataframe[['nwm_percentiles']], dataframe['usgs_percentiles']
            #fit regression model
            model.fit(X, y)
            #calculate R-squared of regression model
            r_squared = model.score(X, y)
            percentile_rsqs.append(r_squared)
        except:
            r_squared = "nan"
            percentile_rsqs.append(r_squared)

        gage_ids.append(gage_id)
        comids.append(rivid)
        lats.append(lat)
        lons.append(lon)
        # except:
        #     print("didn't work for {0}".format(gage_id))
        #     pass

    output_data = {
        'gage': gages,

        'comid': comids,

        'lat': lats,

        'lon':lons,

        'streamflow_nse' : streamflow_nses,

        'percentile_nse' : percentile_nses,

        'streamflow_rsquared': streamflow_rsqs,

        'percentile_rsquared': percentile_rsqs

    }

    # Create dataframe
    dataframe = pd.DataFrame(output_data,columns=['gage','comid','lat','lon','streamflow_nse',
                                                  'percentile_nse','streamflow_rsquared','percentile_rsquared'])

    # output the results in csv file
    output_csv = "/data/2023_WRAP_Streamflow_APT/Tables/GageLoc.csv"
    dataframe.to_csv(output_csv)
