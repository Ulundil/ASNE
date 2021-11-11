# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:29:35 2021

@author: luuk-
"""

import ast
import numpy as np
import os
import pandas as pd
import time
from heapq import nlargest

dumping_factor = 0.85



# prod = np.dot(embeddings[0], embeddings[1])

def load_data():
    file_path = "ASNE_input.pkl"
    if os.path.exists(file_path):
        print("Loading existing file!")
        user_df = pd.read_pickle(file_path)
        print("Loaded!")
    else:
        print("No data file found!")
        return pd.DataFrame()
    
    file = open("model.json", 'r')
    embeddings = ast.literal_eval(file.read())
    user_df['Embedding'] = embeddings
    
    file = open("node_neighbors_map.json", "r")
    neighbours = dict([int(key), value] for key, value in ast.literal_eval(file.read()).items())
    print(type(neighbours))
    print(len(neighbours))
    user_df['Neighbours'] = pd.Series(neighbours)
    user_df['Neighbours'] = [ [] if x is np.NaN else x for x in user_df['Neighbours'] ]
    for index, row in user_df.iterrows():
        lst = row['Neighbours']
        for index, item in enumerate(lst):
            lst[index] = user_df.iloc[item]['ID']
        row.at['neighbours'] = lst
    
    return user_df

def get_strength(user_i, user_j):
    # print(user_j['ID'], type(user_j['ID']), type(user_i['Following list']))
    # print(user_j)
    if user_j['ID'] in user_i['Following list']:
        return np.dot(user_i['Embedding'], user_j['Embedding'])
    else:
        return 0

def sne_rank(df, dump):
    st = time.time()
    sum_w = 0
    sum_m_dict = {}
    sum_neigh_dict = {}
    rank_values = {}
    user_list = df['ID'].tolist()
    for index, row in df.iterrows():
        sum_m = 0
        user_i = row['ID']
        for i2, r2 in df.iterrows():
            user_j = r2['ID']
            if user_i != user_j:
                # print(row)
                # print("=============================================================================")
                # print(row, r2)
                sum_m += get_strength(row, r2)
        sum_m_dict[user_i] = sum_m
        sum_w += sum_m
        
        neigh_list = row['Neighbours']
        sum_w_neigh = 0
        for user_j in neigh_list:
            # j_idx = df.index[df['ID'] == user_j]
            # # print(j_idx)
            # j_row = df.iloc[j_idx]
            j_row = df[df['ID'] == user_j].iloc[0]
            sum_w_neigh += get_strength(row, j_row)
        sum_neigh_dict[user_i] = sum_w_neigh
    print(f"Done with part 1 ({time.time() - st} seconds)")
    st = time.time()
    for index, row in df.iterrows():
        user = row['ID']
        if sum_w > 0 and sum_neigh_dict[user] > 0:
            rank_value = ((1-dump) * (sum_m_dict[user] / sum_w)) + ((dump) * (sum_m_dict[user] / sum_neigh_dict[user]))
        else:
            rank_value = 0
        if rank_value < 0.01:
            print(f"""Rank value: {rank_value}\n
                      sum_m_dict: {sum_m_dict[user]}\n
                      sum_w: {sum_w}\n
                      sum_neigh_dict: {sum_neigh_dict[user]}\n
                      User: {user}\n
                      Index: {index}""")
        rank_values[user] = rank_value
    
    print(f"Done with part 2 ({time.time() - st} seconds)")
    return rank_values


st = time.time()
df = load_data()
ranks = sne_rank(df, dumping_factor)
print(f"Total time to determine SNERank: {time.time() - st}")
top_5 = nlargest(5, ranks, key = ranks.get)
print("Top 5 highest-ranking users:")
for val in top_5:
    name = df[df['ID'] == val].Name.values[0]
    followers = df[df['ID'] == val]['Followers count'].values[0]
    print(f"{name} | {val} | {followers} followers (rank: {ranks[val]})")