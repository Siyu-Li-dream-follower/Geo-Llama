 # Geolife stay point detection

import pandas as pd
import numpy as np
import os
import sys
from os.path import join, exists
from geopy import distance
import geopandas as gpd
from shapely.geometry import Point
import argparse
import time

class bounding_box:
    def __init__(self, _lat_min, _lon_min,_lat_max,_lon_max):
        self.lat_min = _lat_min
        self.lon_min = _lon_min
        self.lat_max = _lat_max
        self.lon_max = _lon_max

class stat_collector:
    def __init__(self):
        self.parquet_file_count=0
        self.data_record_count = 0
        self.memory_usage_in_GB = 0		
        self.unique_device_count = 0
        self.avg_pos_acc = 0
        self.starting_time = time.process_time()
        self.elapsed_time = time.process_time()
        self.unique_geohash_count = 0
        self.avg_records_per_device = 0

def save_df_to_file(df, output_path):
    save_format = output_path.split('.')[-1]
    if save_format == 'csv':
        df.to_csv(output_path, index=False)
    elif save_format == 'h5':
        df.to_hdf(output_path, 'data')
    else:
        sys.exit('Save file format not supported!')


def read_df_from_file(input_path):
    input_format = input_path.split('.')[-1]
    if input_format == 'csv':
        df = pd.read_csv(input_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    elif input_format == 'h5':
        df = pd.read_hdf(input_path, 'data')
    elif input_format == 'txt':
        df = pd.read_csv(input_path, sep=' ',names=['user_id', 'lat', 'long', 'arrival_time', 'leave_time'])
        #print(df.head(n=10).to_string(index=False))
    else:
        sys.exit('Input file format not supported!')
    return df


def get_single_traj(filepath):
    # Get dataframe of 1 trajectory from 1 file
    header = ['lat', 'long', 'useless', 'alt', 'timestamp', 'date', 'time']
    df = pd.read_csv(filepath, skiprows=6, header=None, names=header)
    df['datetime'] = df.apply(lambda x: pd.to_datetime(x.date + ' ' + x.time), axis=1)
    df['transport'] = ''
    df = df.drop(['useless', 'timestamp', 'date', 'time'], axis=1)
    return df


def get_trans_mode(filepath):
    # Get the dataframe of transportation mode from a single file containing the modes of multiple trajectories.
    df = pd.read_csv(filepath, sep='\t')
    df['start'] = pd.to_datetime(df['Start Time'])
    df['end'] = pd.to_datetime(df['End Time'])
    df['transport'] = df['Transportation Mode']
    df = df.drop(['Start Time', 'End Time', 'Transportation Mode'], axis=1)
    return df


def merge_geolife_data(input_path, output_path=None):
    '''
    Merge the GeoLife data into a single file
    Args:
        input_path: the FOLDER contains all geolife files
        output_path: the file path to save
    Returns:
        A dataframe of all trajectories
    '''
    USER_SAMPLE = [x.zfill(3) for x in list(map(str, list(range(0, 181))))]
    HEADERS = ['traj_id', 'user_id', 'lat', 'long', 'alt', 'datetime', 'transport']
    list_df = []
    for user_dir in USER_SAMPLE:
        traj_dir = join(input_path, user_dir, 'Trajectory')
        print(traj_dir)
        traj_files = os.listdir(traj_dir)

        user_id = user_dir.split('.')[0]
        list_traj = []
        for traj in traj_files:
            traj_id = traj.split('.')[0]
            df = get_single_traj(join(traj_dir, traj))
            df['user_id'] = user_id
            df['traj_id'] = user_id + '_' + traj_id
            list_traj.append(df)
        df_traj = pd.concat(list_traj)
        if exists(join(input_path, user_dir, 'labels.txt')):
            df_label = get_trans_mode(join(input_path, user_dir, 'labels.txt'))
            for idx, row in df_label.iterrows():
                df_traj.loc[(df_traj['datetime'] >= row['start']) & (df_traj['datetime'] <= row['end'])
                , 'transport'] = row['transport']
        list_df.append(df_traj)
    df_sample = pd.concat(list_df)
    df_sample = df_sample.reset_index()
    df_sample = df_sample[HEADERS]

    # We limit the area to Beijing to match our POI data.
    
    ### Large area entire Beijing, used in the paper ###
    df_sample = df_sample[(df_sample['lat'] >= 39.45) & (df_sample['lat'] <= 41.05)
                          & (df_sample['long'] >= 115.41) & (df_sample['long'] <= 117.5)]
    ### lat difference: 1.6
    ### long difference: 2.09
    
    ### Small area not used in the paper ###
    # df_sample = df_sample[(df_sample['lat'] >= 39.75) & (df_sample['lat'] <= 40.1)
    #                       & (df_sample['long'] >= 116.2) & (df_sample['long'] <= 116.55)]
    ### lat difference: 0.35
    ### long difference: 0.35
    
    if output_path:
        save_df_to_file(df_sample, output_path)
    return df_sample


def get_stay_point(df_traj, DIST_THRES, TIME_THRES):
    '''
    input: a single trajectory dataframe
    process: we set the time threshold and utilize a spatial buffer to obtain stay point with less computation cost
    return: a sequence of stay points
    '''
    geometry = [Point(xy) for xy in zip(df_traj['long'], df_traj['lat'])]
    crs = {'init': 'epsg:4326'}
    gdf_traj = gpd.GeoDataFrame(df_traj, crs=crs, geometry=geometry)
    df_stay = pd.DataFrame([], columns=['traj_id', 'user_id', 'lat', 'long', 'arrival_time', 'leave_time', 'stay_time'])
    i = 0

    while i < len(df_traj) - 1:
        if distance.distance((df_traj.iloc[i]['lat'], df_traj.iloc[i]['long']),
                             (df_traj.iloc[i + 1]['lat'], df_traj.iloc[i + 1]['long'])) > DIST_THRES:
            i = i + 1
            continue
        buffer = gdf_traj.iloc[i:i + 1].buffer(DIST_THRES)
        df_neighbor = gdf_traj.iloc[i + 1:].copy().reset_index(drop=True)
        df_candidate = df_neighbor[df_neighbor['geometry'].intersects(buffer.unary_union)]
        df_first = df_candidate[(df_candidate.index - df_neighbor[0:len(df_candidate)].index) == 0]
        time_delta = (df_first.iloc[-1]['datetime'] - df_traj.iloc[i]['datetime']).total_seconds()
        if time_delta > TIME_THRES:
            df_first = pd.concat([df_first, df_traj.iloc[i:i + 1]])
            staypoint = {}
            staypoint['traj_id'] = df_traj.iloc[i]['traj_id']
            staypoint['user_id'] = df_traj.iloc[i]['user_id']
            staypoint['lat'] = df_first['lat'].mean()
            staypoint['long'] = df_first['long'].mean()
            staypoint['arrival_time'] = df_traj.iloc[i]['datetime']
            staypoint['leave_time'] = df_first.iloc[-2]['datetime']
            staypoint['stay_time'] = time_delta
            df_stay = pd.concat([df_stay, pd.DataFrame([staypoint])], ignore_index=True)
            i = i + len(df_first)
        i = i + 1

    start = {'arrival_time': df_traj.iloc[0]['datetime'], 'leave_time': df_traj.iloc[0]['datetime'], 'stay_time': 0}
    end = {'arrival_time': df_traj.iloc[-1]['datetime'], 'leave_time': df_traj.iloc[-1]['datetime'], 'stay_time': 0}
    df_stay = pd.concat([df_stay, pd.DataFrame([dict(df_traj.iloc[0][['traj_id', 'user_id', 'lat', 'long']].to_dict(), **start)])], ignore_index=True)
    df_stay = pd.concat([df_stay, pd.DataFrame([dict(df_traj.iloc[-1][['traj_id', 'user_id', 'lat', 'long']].to_dict(), **end)])], ignore_index=True)
    df_stay = df_stay.sort_values(by='arrival_time', ascending=True)
    return df_stay

def spd(df, DIST_THRES, TIME_THRES, output_path=None):
    '''
    DIST_THRES is degree, TIME_THRES is second
    '''
    list_df = []
    for name, group in df.groupby(by='traj_id'):
        df_stay = get_stay_point(group.copy(), DIST_THRES, TIME_THRES)
        list_df.append(df_stay)
    df_spd = pd.concat(list_df)
    if output_path:
        save_df_to_file(df_spd, output_path)
    print('Finished stay point detection!')
    return df_spd

def main(args):
    func = args.function
    input_path = args.input_path
    output_path = args.output_path

    if func == 'merge_geolife_data':
        merge_geolife_data(input_path, output_path)

    elif func == 'spd':
        s_threshold = args.s_thresh
        t_threshold = args.t_thresh
        df_clean = read_df_from_file(input_path)
        spd(df_clean, s_threshold, t_threshold, output_path)

if __name__ == '__main__':
    # task = 'merge_geolife_data'
    # input_path = "Data"
    # output_path = "geo_data.h5"
    
    task = 'spd'
    input_path = "geo_data.h5"
    output_path = "geo_spd.h5"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default=task, type=str, help='choose from "merge_geolife_data", '
                                                                      '"clean", "spd", "augmentation"')
    parser.add_argument('--input_path', default=input_path, type=str,
                        help='the path of the input data file/folder')
    parser.add_argument('--output_path', default=output_path, type=str,
                        help='the path of the output data file (.csv or .h5)')

    # args for spd
    parser.add_argument('--s_thresh', default=0.0015, type=float, help='space threshold') # 0.01 degree is about 1.1 km
    parser.add_argument('--t_thresh', default=20 * 60, type=int, help='time threshold')

    args = parser.parse_args()
    main(args)
