# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:50:11 2021

@author: Gebruiker
"""
# Import libraries
import pandas as pd
import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
import random
import numpy as np
import re
from os import listdir
import math

# Set the week numbers to use
weeks = [1]

# Set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def most_frequent(lst):
    return max(set(lst), key = lst.count)

def find_user_stances(raw_data):
    stance_dict = {}
    for user in raw_data.itertuples():
        if user.id_str in stance_dict:
            stance_dict[user.id_str].append(user.label_social_distancing)
        else:
            stance_dict[user.id_str] = [user.label_social_distancing]
    
    unique_users = raw_data.drop_duplicates(subset = "id_str").reset_index()
    
    for key in stance_dict:
        stance_dict[key] = most_frequent(stance_dict[key])    
    
    unique_users['User stance'] = unique_users['id_str'].map(stance_dict)
    return unique_users

def split(a, n): # Splits list a into n approximately equal parts and returns them as a list of lists
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def bag_of_words(text): # Extracts TF-IDF from a series or list
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    # text_counts= cv.fit_transform(text)
    tf=TfidfVectorizer(lowercase = True, stop_words = "english", ngram_range = (1, 1), tokenizer = token.tokenize)
    text_tf= tf.fit_transform(text)
    return text_tf
    
def create_attr_info(index, row):
    idx_str = str(index+1)
    stance = row['Stance']
    if stance == "SUPPORTS":
        stance_str = "2_1"
    else:
        stance_str = "2_0"
    text_list = row['TF-IDF Text'].data
    text_str = ""
    for i in range(len(text_list)):
        text_str += str(i+3)
        text_str += "_"
        text_str += str(text_list[i]) + " "
    return " ".join([idx_str, stance_str, text_str])

def create_doublelink(df):
    edges = []
    name_to_index = {}
    file_path = 'data/doublelink.EDGELIST'
    file_path_json = 'data/names_to_index.json'
    file = open(file_path, "w")
    file.truncate()
    file.close
    st = time.time()
    for index, row in df.iterrows():
        curr_list = row['Following list']
        name_to_index[index+1] = (row['Name'], row['ID'], index+1)
        # print(curr_list)
        if index%1000 == 0:
            print(index, f"{time.time() - st} seconds")
            st = time.time()
        if isinstance(curr_list, list):
            for name in curr_list:
                # print(row['Name'], name, len(curr_list), curr_list.index(name))
                name_idx = df[df["ID"] == name].index.values[0]
                # print(index, name_idx)
                edges.append((index+1, name_idx+1))
                with open(file_path, 'a') as file:
                    string = f"{index+1} {name_idx+1}\n"
                    if string == "":
                        print(name)
                    else:
                        file.write(string)
            # break
    js = json.dumps(name_to_index)
    f = open(file_path_json, 'w')
    f.write(js)
    f.close()
    return (edges, name_to_index)
    
def create_test_file(df, edges):
    num_test = min(43816, len(edges))
    num_links = int(num_test/2) # Half of the test cases are linked nodes, half are unlinked nodes
    num_non_links = num_test - num_links
    # Get num_link random numbers between 1 and 43816
    links = random.sample(range(num_test), num_links)
    non_links = []
    count = 0
    df_size = df.shape[0]
    while count < num_non_links:
        idx_1 = random.randrange(df_size)
        idx_2 = random.randrange(df_size)
        if idx_1 != idx_2:
            if not ((idx_1+1, idx_2+1) in edges or (idx_2+1, idx_1+1) in edges):
                non_links.append((idx_1+1, idx_2+1))
                count += 1
    # print(len(non_links))
    
    test_links = []
    for i in range(len(links)):
        idx = links[i]
        test_links.append(" ".join([str(edges[idx][0]), str(edges[idx][1]), str(1)]))
    for i in range(len(non_links)):
        test_links.append(" ".join([str(non_links[i][0]), str(non_links[i][1]), str(0)]))
    
    file_path = 'data/test_pairs.txt'
    file = open(file_path, "w")
    file.truncate()
    file.close
    with open(file_path, mode = "w") as file:
        for link in test_links:
            file.write(f"{link}\n")
    return (links, non_links, test_links)

def create_gephi_csv(df, edges):
    edge_csv_list = []
    for edge in edges:
        edge_csv_list.append(f"{edge[0]};{edge[1]}")
    file_path = 'data/gephi_graph.csv'
    file = open(file_path, "w")
    file.truncate()
    file.close
    with open(file_path, mode = "w") as file:
        for edge in edge_csv_list:
            file.write(f"{edge}\n")
    return edge_csv_list
    

def swap_dict(old_dict):
    new_dict = {}
    for key, value in old_dict.items():
        for string in value:
            new_dict.setdefault(string, []).append(key)
    return new_dict

def remove_values(dct, lst):
    # Remove values in lst from the dct values
    dict_keys = list(dct.keys())
    lst = set(lst)
    dict_values = list(dct.values())
    new_values = []
    for values_list in dict_values:
        old_val = len(values_list)
        new_values_list = list(lst & set(values_list))
        new_values.append(new_values_list)
        # print(f"{old_val} | {len(new_values_list)}")
    new_dict = dict(zip(dict_keys, new_values))
    return new_dict

def create_data_file(weeks, tiny_file = False):
    print(f"Week numbers used: {weeks}")
    # Import the data json into a dataframe
    start_time = time.time()
    raw_data = pd.read_json(r"../social-distancing-student.json", lines = True, dtype = 'object')
    raw_data = raw_data.astype({'created_at': 'datetime64'})
    raw_data['full_text'] = raw_data.apply(lambda x: x['full_text'] if x['full_text'] != "" else x['text'], axis = 1)
    # week_data = []
    # for week_num in weeks:
    #     week1_data = raw_data[raw_data['created_at'].dt.isocalendar().week == 40+week_num].copy()
    #     week_data.append(week1_data)
    query = ""
    for week in weeks:
        query += f"created_at.dt.isocalendar().week == {39+week} or "
    query = query[:-4]
    week1_data = raw_data.copy().query(query)
    week1_data.rename(columns = {"id_str": "id_str_tweet"}, inplace = True)
    user_unpacked = pd.json_normalize(week1_data['user'])
    week1_data = pd.concat([week1_data, user_unpacked], axis = 1)
    if tiny_file == True:
        week1_data = week1_data.head(1000)
    end_time = time.time()
    print(f"Time to load the data: {end_time - start_time} seconds")
    st = time.time()
    # test = week1_data[week1_data['id_str'] == '190648628']['text']
    # print(test)
    # return 0
    # Create the user dataframe
    user_df = pd.DataFrame(columns = ['Name', 'ID', 'Stance', 'Followers list', 'Following list'])
    
    # Extract the unique users from the week 1 data
    # unique_users = week1_data.drop_duplicates(subset = "id_str")
    unique_users = find_user_stances(week1_data)
    # Only extract users with 1000 or less followers
    # unique_users = unique_users[unique_users['followers_count'] <= 1000]
    
    # Extract a list of the text of all tweets made by a single user, for all users
    grouped_tweets = week1_data[["id_str", "full_text", 'followers_count']].copy()
    u = grouped_tweets.groupby("id_str")["full_text"].apply(list).reset_index(name = "Tweet text")
    # print(u['Tweet text'].head())
    # print(u['Tweet text'].iloc[0])
    # print(len(u['Tweet text'].iloc[0]))
    # return 0
    u['Tweet text'] = u['Tweet text'].apply(lambda x: [i for i in x if i == i])
    grouped_tweets = pd.merge(grouped_tweets, u, on = "id_str")
    grouped_tweets.drop_duplicates(subset = "id_str", inplace = True)  
    # grouped_tweets = grouped_tweets[grouped_tweets['followers_count'] <= 1000]
    
    # print(unique_users.head())
    
    # Append the information to the user_df
    user_df['Name'] = unique_users['screen_name']
    user_df['ID'] = unique_users['id_str']
    user_df['Stance'] = unique_users['User stance']
    user_df = pd.merge(user_df, grouped_tweets[["Tweet text", "id_str", "followers_count"]], left_on = "ID", right_on = "id_str")
    user_df.drop("id_str", axis = 1, inplace = True)
    user_df.rename(columns = {"followers_count": "Followers count"}, inplace = True)
    print(f"Time to reconfigure all DataFrames: {time.time() - start_time} seconds")
    st = time.time()
    # Read in the followers list for each user and store it in the dataframe
    base_pattern = "\s.+\sfollowers.txt"
    followers_dict = {}
    for index, row in user_df.iterrows():
        followers = []
        for week in weeks:
            file_path = f"../../week{week}_followers/"
            file_list = listdir(file_path)
            pattern = re.compile(row['ID'] + base_pattern)
            # followers_path = f"{file_path + row['ID']} {row['Name']} followers.txt"
            file = list(filter(pattern.search, file_list))
            if len(file) > 0:
                followers_file = open(file_path + file[0], 'r')
                while True:
                    line = followers_file.readline()
                    if not line:
                        break
                    followers.append(line.strip())
                # with open(followers_path) as followers_file:
                #     followers = 
                # followers_dict.update(followers)
        followers_dict[row['ID']] = followers
    print(len(followers_dict))
    # # Remove the hashtags from every name
    # for key in followers_dict:
    #     lst = [e[1:] for e in followers_dict[key]]
    #     followers_dict[key] = lst
    print(f"Time to find all followers: {time.time() - st} seconds")
    st = time.time()
    # Add the followers to the dataframe as well
    user_df["Followers list"] = user_df["ID"].map(followers_dict)
    # print("Done with the followers")
    # Get TF-IDF from each user's tweets
    tfidf_list = []
    for index, row in user_df.iterrows():
        # print(row['Tweet text'])
        if len(row['Tweet text']) > 0:
            matrix = bag_of_words(row['Tweet text'])
        else:
            matrix = []
        tfidf_list.append(matrix)
    
    # Add the TF-IDF list to the dataframe
    user_df["TF-IDF Text"] = tfidf_list
    
    # # Create the attr_info file
    # file_path = 'data/attr_info.txt'
    # file = open(file_path, "w")
    # file.truncate()
    # file.close
    # for index, row in user_df.iterrows():
    #     string = create_attr_info(index, row)
    #     with open(file_path, 'a') as file:
    #         file.write(string + "\n")
    
    end_time = time.time()
    print(f"Time until getting following: {end_time - st} seconds")
    st = time.time()
    # return followers_dict
    # Get all the following relationships
    followers_dict = remove_values(followers_dict, user_df['ID'].to_list())
    print(f"Time to clean followers dict: {time.time() - st} seconds")
    st = time.time()
    following_dict = swap_dict(followers_dict)
    # return followers_dict, following_dict
    print(f"Time to get following: {time.time() - st} seconds")
    # following_dict = {}
    # st = time.time()
    # for index, row in user_df.iterrows():
    #     # if index % 100 == 0:
    #     #     print(f"{index} ({time.time() - st} seconds)")
    #     #     st = time.time()
    #     if index == 10:
    #         break
    #     name = row['Name']
    #     print(row['ID'])
    #     following = [k for k, v in followers_dict.items() if name in v]
    #     following_dict[name] = following
    #     print(following)
        
    # Add the following to the dataframe
    user_df['Following list'] = user_df["ID"].map(following_dict)
    user_df['Following list'] = user_df['Following list'].apply(lambda d: d if isinstance(d, list) else [])
    user_df = user_df[(user_df['Followers list'].map(len) > 0) | (user_df['Following list'].map(len) > 0)].reset_index(drop = True)
    print(user_df.shape)
    # Create the attr_info file
    file_path = 'data/attr_info.txt'
    file = open(file_path, "w")
    file.truncate()
    file.close
    for index, row in user_df.iterrows():
        string = create_attr_info(index, row)
        with open(file_path, 'a') as file:
            file.write(string + "\n")
    
    # Save the dataframe
    user_df.to_pickle("ASNE_input.pkl")
    # user_df.to_csv("ASNE_input.csv")
    
    end_time = time.time()
    print(f"Time to create ASNE_input file: {end_time - start_time} seconds")
    return user_df

def main():
    start_time = time.time()
    file_path = "ASNE_input.pkl"
    if os.path.exists(file_path):
        print("Loading existing file!")
        user_df = pd.read_pickle(file_path)
        print("Loaded!")
    else:
        print("Creating new data file!")
        user_df = create_data_file(weeks, True)
    st = time.time()
    edges, name_to_index = create_doublelink(user_df)
    print(f"Time to create edge list: {time.time() - st} seconds")
    st = time.time()
    links, non_links, test_links = create_test_file(user_df, edges)
    print(f"Time to create test file: {time.time() - st} seconds")
    st = time.time()
    gephi_csv_list = create_gephi_csv(user_df, edges)
    print(f"Time to create gephi csv: {time.time() - st} seconds")
    end_time = time.time()
    print(f"Time to run the entire program: {end_time - start_time} seconds")
    return locals()


if __name__ == "__main__":
    locals().update(main())