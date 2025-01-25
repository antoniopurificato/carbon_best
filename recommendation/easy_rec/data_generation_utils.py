from multiprocessing.sharedctypes import Value
import pickle as pkl
import os
import json
#import wget
import torch
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime
from scipy import stats

import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from easy_data.data import split_data


def preprocess_dataset(name, 
                       data_folder="../data/raw", 
                       min_rating=None, 
                       min_items_per_user=0, 
                       min_users_per_item=0, 
                       densify_index = True, 
                       split_method="leave_n_out", 
                       split_keys={"sid":["train_sid","val_sid","test_sid"],"timestamp":["train_timestamp","val_timestamp","test_timestamp"],"rating":["train_rating","val_rating","test_rating"]}, 
                       test_sizes=[1,1], 
                       random_state=None, 
                       del_after_split=True, 
                       **kwargs):
    '''
    Preprocesses the dataset.

    Args:
        name (str): Name of the dataset.
        data_folder (str, optional): Path to the dataset folder. Defaults to "../data/raw".
        min_rating (float, optional): Minimum rating value to filter the dataset. Defaults to None.
        min_items_per_user (int, optional): Minimum number of items per user to keep. Defaults to 0.
        min_users_per_item (int, optional): Minimum number of users per item to keep. Defaults to 0.
        densify_index (bool, optional): Whether to densify the index. Defaults to True.
        split_method (str, optional): Splitting method for train/validation/test. Defaults to "leave_n_out".
        split_keys (dict, optional): Dictionary specifying the keys to split and their corresponding new keys. Defaults to {"sid":["train_sid","val_sid","test_sid"],"timestamp":["train_timestamp","val_timestamp","test_timestamp"],"rating":["train_rating","val_rating","test_rating"]}.
        test_sizes (list, optional): List of test set sizes for each split key. Defaults to [1,1].
        random_state (int, optional): Seed for random number generation. Defaults to None.
        del_after_split (bool, optional): Whether to delete the original split key from the data. Defaults to True.

    Returns:
        Tuple: Preprocessed data and mapping information.
    '''
    
    #TODO: variable split dataset? OR move train-test split to basic easy_data?
    # if split_method=="leave_n_out" and min_items_per_user<np.sum(test_sizes):
    #     raise ValueError('Need at least 3+test_num_samples ratings per user for input, train, validation and test: min_items_per_user --> 3+test_num_samples')

    dataset_raw_folder = os.path.join(data_folder, name)

    #start to preprocess the dataset
    maybe_preprocess_raw_dataset(dataset_raw_folder, name)  #check if ratings exists, otherwise preprocess

    df = load_ratings_df(dataset_raw_folder, name)

    # Filter out users/items with less than min_rating ratings
    df = filter_ratings(df, min_rating)
    df = filter_by_frequence(df, min_items_per_user, min_users_per_item)
    if densify_index:
        df, maps = densify_index_method(df) #TODO: add variable densify vars

    data = df_to_sequences(df)

    data = split_rec_data(data, split_method, split_keys, test_sizes, random_state=random_state, del_after_split=del_after_split, **kwargs)

    #TODO: controllare stats
    #print_stats(data, True)
    
    # for k,v in dataset.items():
    #     pass #TODO: what did you have in mind?

    # aggregate all keys in a single list
    # for key in dataset.keys():
    #     dataset[key] = list(dataset[key].values())

    return data, maps

# TODO: check items in train/val/test: already seen in train?


def maybe_preprocess_raw_dataset(dataset_raw_folder, dataset_name):
    '''
    Checks if the raw dataset exists and preprocesses it if necessary.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset.
    '''
    #print(os.path.isdir(dataset_raw_folder), all(os.path.isfile(os.path.join(dataset_raw_folder,filename)) for filename in get_rating_files_per_dataset(dataset_name)))
    #print(dataset_raw_folder, dataset_name)
    if os.path.isdir(dataset_raw_folder) and all(os.path.isfile(os.path.join(dataset_raw_folder,filename)) for filename in get_rating_files_per_dataset(dataset_name)):
        print('Ratings data already exists. Skip pre-processing')
        return
    else:
        specific_preprocess(dataset_raw_folder, dataset_name)


def get_rating_files_per_dataset(dataset_name):
    '''
    Returns a list of rating files corresponding to the given dataset name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        List: List of rating files.
    '''
    if dataset_name == "ml-1m":
        return ['ratings.dat']
    elif dataset_name == "ml-100k":
        return ['u.data']
    elif dataset_name == "ml-20m":
        return ['ratings.csv']
    elif dataset_name == "steam":
        return ['steam.csv']
    elif dataset_name == "amazon_beauty":
        return ['All_Beauty.csv']
    elif dataset_name == "amazon_videogames":
        return ['Video_Games.csv']
    elif dataset_name == "amazon_toys":
        return ['Toys_and_Games.csv']
    elif dataset_name == "amazon_cds":
        return ['CDs_and_Vinyl.csv']
    elif dataset_name == "amazon_music":
        return ['Digital_Music.csv']
    elif dataset_name == "amazon_books":
        return ['Books.csv']
    elif dataset_name == "foursquare-nyc":
        return ['dataset_TSMC2014_NYC.txt']
    elif dataset_name == "foursquare-tky":
        return ['dataset_TSMC2014_TKY.txt']
    elif dataset_name == "behance":
        return ['behance.csv']
    elif dataset_name == "yelp":
        return ['yelp.csv']
    else:
        raise NotImplementedError


def specific_preprocess(dataset_raw_folder, dataset_name): 
    '''
    Performs dataset-specific preprocessing.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset.
    '''
    
    #TODO filippo check the code for ML-20M, Gowalla
    # For the "steam" dataset
    if dataset_name == "steam":
        # File path for the Steam dataset
        file_path = os.path.join(dataset_raw_folder, 'steam.json')  # IT'S NOT A JSON... (NOR jsonl: single quotes instead of doubles)
        all_reviews = []
        # Read and process each line in the file
        with open(file_path, "r") as f:
            for line in f:
                # Convert each line to a dictionary using literal_eval
                line_dict = literal_eval(line)
                user_id = line_dict['username']
                # Extract relevant information from each review
                #for review_dict in line_dict['reviews']:
                item_id = line_dict['product_id']
                #rating = review_dict['recommend'] * 1
                rating = 3  #TODO: check if it's correct
                timestamp = line_dict['date']#[7:-1]  # removing "Posted " and "."
                try:
                    # Convert the timestamp to a Unix timestamp
                    timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(timestamp, "%Y-%m-%d"))
                except ValueError:
                    timestamp = -1
                timestamp = int(timestamp)
                
                all_reviews.append((user_id, item_id, rating, timestamp))

        # Convert the processed data to a DataFrame and save it as a CSV file
        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, 'steam.csv'), header=False, index=False)
    elif dataset_name == "yelp":
        # File path for the Yelp dataset
        file_path = os.path.join(dataset_raw_folder, 'yelp_academic_dataset_review.json')
        all_reviews = []
        # Read and process each line in the file
        with open(file_path, "r") as f:
            for line in f:
                # Convert each line to a dictionary using literal_eval
                line_dict = literal_eval(line)
                user_id = line_dict['user_id']
                # Extract relevant information from each review
                item_id = line_dict['business_id']
                rating = line_dict['stars']
                timestamp = line_dict['date']
                try:
                    # Convert the timestamp to a Unix timestamp
                    timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    timestamp = -1
                timestamp = int(timestamp)
                
                all_reviews.append((user_id, item_id, rating, timestamp))

        # Convert the processed data to a DataFrame and save it as a CSV file
        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, 'yelp.csv'), header=False, index=False)
    elif dataset_name == "behance":
        file_path = os.path.join(dataset_raw_folder, 'behance.txt')  # IT'S NOT A JSON... (NOR jsonl: single quotes instead of doubles)
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                user_id, item_id, timestamp = line.strip().split(" ")
                rating = 3 #TODO: check if it's correct
                timestamp = int(timestamp)
                all_reviews.append((user_id, item_id, rating, timestamp))
        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, 'behance.csv'), header=False, index=False)

    # For Amazon datasets
    elif "amazon" in dataset_name:
        # Mapping dataset_name to the original file name
        if dataset_name == "amazon_beauty":
            orig_file_name = 'All_Beauty'
        elif dataset_name == "amazon_videogames":
            orig_file_name = 'Video_Games'
        elif dataset_name == "amazon_toys":
            orig_file_name = 'Toys_and_Games'
        elif dataset_name == "amazon_cds":
            orig_file_name = 'CDs_and_Vinyl'
        elif dataset_name == "amazon_music":
            orig_file_name = 'Digital_Music'
        elif dataset_name == "amazon_books":
            orig_file_name = 'Books'

        # File path for the Amazon dataset
        file_path = os.path.join(dataset_raw_folder, orig_file_name + '.json')  # IT'S NOT A JSON... (NOR jsonl: single quotes instead of doubles)
        all_reviews = []
        # Read and process each line in the file
        with open(file_path, "r") as f:
            for line in f:
                # Replace "verified" values to match Python's True/False
                line = line.replace('"verified": true,', '"verified": True,').replace('"verified": false,', '"verified": False,')
                # Convert each line to a dictionary using literal_eval
                line_dict = literal_eval(line)
                user_id = line_dict['reviewerID']
                item_id = line_dict['asin']
                rating = float(line_dict['overall'])
                timestamp = line_dict['unixReviewTime']
                all_reviews.append((user_id, item_id, rating, timestamp))

        # Convert the processed data to a DataFrame and save it as a CSV file
        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, orig_file_name + '.csv'), header=False, index=False)
    
    # If the dataset_name is neither "steam" nor an Amazon dataset, raise NotImplementedError
    else:
        print(dataset_name)
        raise NotImplementedError


def load_ratings_df(dataset_raw_folder, dataset_name):
    '''
    Loads the ratings DataFrame from the dataset.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset.

    Returns:
        pd.DataFrame: Ratings DataFrame.
    '''

    if dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder,'ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder,'u.data')
        df = pd.read_csv(file_path, sep='\t', header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "ml-20m":
        file_path = os.path.join(dataset_raw_folder,'ratings.csv')
        df = pd.read_csv(file_path, sep=',', header=0, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif "amazon" in dataset_name or dataset_name=="steam" or dataset_name=="behance" or dataset_name=="yelp":
        if dataset_name == "steam":
            orig_file_name = 'steam'
        elif dataset_name == "yelp":
            orig_file_name = 'yelp'
        elif dataset_name == "behance":
            orig_file_name = 'behance'
        elif dataset_name == "amazon_beauty":
            orig_file_name = 'All_Beauty'
        elif dataset_name == "amazon_videogames":
            orig_file_name = 'Video_Games'
        elif dataset_name == "amazon_toys":
            orig_file_name = 'Toys_and_Games'
        elif dataset_name == "amazon_cds":
            orig_file_name = 'CDs_and_Vinyl'
        elif dataset_name == "amazon_music":
            orig_file_name = 'Digital_Music'
        elif dataset_name == "amazon_books":
            orig_file_name = 'Books'
        file_path = os.path.join(dataset_raw_folder, orig_file_name + '.csv')
        df = pd.read_csv(file_path, header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif "foursquare" in dataset_name:
        if dataset_name == "foursquare-nyc":
            filename = 'dataset_TSMC2014_NYC.txt'
        elif dataset_name == "foursquare-tky":
            filename = 'dataset_TSMC2014_TKY.txt'
        file_path = os.path.join(dataset_raw_folder,filename)
        df = pd.read_csv(file_path, sep='\t', header=None, encoding='latin-1', engine="python")
        df.columns = ['uid', 'sid', "s_cat", "s_cat_name", "latitude", "longitude", "timezone_offset", "UTC_time"]
        df["rating"] = 1 #there are no ratings
        df["timestamp"] = df["UTC_time"].apply(lambda x: datetime.datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").timestamp())
        return df
    else:
        raise NotImplementedError
    

#implicit = don't use ratings
#explicit = keep ratings
def filter_ratings(df, min_rating):
    '''
    Filters ratings DataFrame based on the minimum rating.

    Args:
        df (pd.DataFrame): Ratings DataFrame.
        min_rating (float): Minimum rating value.

    Returns:
        pd.DataFrame: Filtered ratings DataFrame.
    '''
    if min_rating is not None:
        #df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df[df['rating'] >= min_rating]
    return df


def filter_by_frequence(df, min_items_per_user, min_users_per_item):
    '''
    Filters DataFrame based on minimum items per user and minimum users per item.

    Args:
        df (pd.DataFrame): DataFrame.
        min_items_per_user (int): Minimum number of items per user.
        min_users_per_item (int): Minimum number of users per item.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    '''
    if min_users_per_item > 0:
        print('Filtering by minimum number of users per item:',min_users_per_item)
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_users_per_item]
        df = df[df['sid'].isin(good_items)]

    if min_items_per_user > 0:
        print('Filtering by minimum number of items per user:',min_items_per_user)
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_items_per_user]
        df = df[df['uid'].isin(good_users)]

    return df


def densify_index_method(df, vars=["uid", "sid"]):
    '''
    Densifies index in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame.
        vars (list): List of variables.

    Returns:
        pd.DataFrame: DataFrame with densified index.
        dict: Mapping between original and densified values.
    '''


    # Print a message indicating that the index densification process is starting
    print('Densifying index')

    # Create a dictionary to store the mapping between original values and densified values
    maps = {}

    # Iterate over variable names and corresponding map names
    for var_name in vars:
        map_name = var_name# + "_map"
        # Create a mapping from original values to dense indices
        maps[map_name] = {u: i + 1 for i, u in enumerate(set(df[var_name]))}  # Probably not a great way to name maps

        # Update the dataframe column with densified indices using the created mapping
        df[var_name] = df[var_name].map(maps[map_name])

    # Return the updated dataframe and the maps dictionary
    return df, maps


def df_to_sequences(df, keep_vars=["uid"], seq_vars=["sid", "rating", "timestamp"], user_var="uid", time_var="timestamp"):
    '''
    Converts DataFrame to sequences.

    Args:
        df (pd.DataFrame): DataFrame.
        keep_vars (list): Variables to keep.
        seq_vars (list): Variables to be included in sequences.
        user_var (str): User variable.
        time_var (str): Timestamp variable.

    Returns:
        dict: Dictionary of sequences.
    '''


    df_group_by_user = df.groupby(user_var)

    data = {}

    for var in seq_vars: #order by time_var
        data[var] = df_group_by_user.apply(lambda d: list(d.sort_values(by=time_var)[var])).values
        
    for var in keep_vars: #keep as they are
        data[var] = df_group_by_user.apply(lambda d: list(d[var])[0]).values #0 cause variable should be the same for every entry

    return data

def print_stats(complete_set,keep_time):  #TODO CHECK WE DONT DO IT HERE BUT IN THE .IPYNB
    print("NUM USERS:",len(complete_set))

    if keep_time:
        print("NUM ITEMS:",len(set(np.concatenate([seq for u,(seq,times) in complete_set.items()]))))
    else:
        print("NUM ITEMS:",len(set(np.concatenate([seq for u,seq in complete_set.items()]))))

    if keep_time:
        lens = [len(seq) for u,(seq,times) in complete_set.items()]
    else:
        lens = [len(seq) for u,seq in complete_set.items()]
    print("AVERAGE LEN:",np.mean(lens))
    print("MEDIAN LEN:",np.median(lens))
    print("MODE LEN:",stats.mode(lens))
    print("STD LEN:",np.std(lens))
    print("MIN/MAX LEN:",np.min(lens),np.max(lens))

    if keep_time:
        print("NUM INTERACTIONS:",np.sum([len(seq) for u,(seq,times) in complete_set.items()]))
    else:
        print("NUM INTERACTIONS:",np.sum([len(seq) for u,seq in complete_set.items()]))

def split_rec_data(data, split_method, split_keys, test_sizes, **kwargs):
    '''
    Splits the dataset based on the specified split method and keys.

    Args:
        data (dict): Input dataset.
        split_method (str): Splitting method.
        split_keys (dict): Splitting keys.
        test_sizes (list): Sizes of the test sets.
        kwargs: Additional arguments.

    Returns:
        dict: Split dataset.
    '''
    print('Splitting:',split_method)
    if split_method == 'leave_n_out':
        for orig_key,new_keys in split_keys.items(): #TODO? float test_sizes
            while len(test_sizes)<len(new_keys):
                test_sizes.append(0)
            end_ids = np.array([len(seq) for seq in data[orig_key]])
            previous_key = orig_key
            for new_key,test_size in zip(new_keys[::-1],test_sizes[::-1]): #last key is the one with the most recent interactions
                end_ids -= test_size
                data[new_key] = np.array([seq[:end_ids[i]] for i,seq in enumerate(data[previous_key])], dtype=object)
                previous_key = new_key
            if "del_after_split" in kwargs and kwargs["del_after_split"]:
                del data[orig_key]
    # elif split_method == 'hold_out':
    #     data = split_data(data, split_keys, test_sizes, **kwargs) #split per user
    # #TODO: split by interactions
    #     #...
    else:
        raise NotImplementedError
    return data



# GRAPH
def transform_to_graph(data, keys=["uid", "sid"], max_ids=None): #TODO DELETE BEFORE PUSHING
    # max_ids = max id per each key in keys
    if max_ids is None: #Must be defined
        raise AttributeError("len_keys must be defined")
    if len(keys) != len(max_ids):
        raise AttributeError("len_keys must be equal to len(len_keys)")
    
    # Create the edge index
    edge_index = []
    
    # Iterate over the data, concatenating each sample
    for edge in zip(*[data[key] for key in keys]):
        edges = [edge].copy()
        redo = True
        while redo:
            redo = False
            new_edges = []
            for edge in edges:
                for i,object in enumerate(edge):
                    if isinstance(object,list) or isinstance(object,tuple): #TODO: if iterable
                        for obj in object:
                            new_edges.append((*edge[:i],obj, *edge[i+1:]))
                        redo = True
                        break
            if redo:
                edges = new_edges
        edge_index.extend(edges)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)#.t()#.contiguous()

    return edge_index_tensor
    #TODO: check if this variation improves the efficiency
        #edge_index.extend(torch.LongTensor(edges))
    #edge_index = torch.stack(edge_index)

    #return edge_index
    
def get_popularity_items(data: dict, num_items: int) -> torch.Tensor:
    popularity = {}
    for sample in data:
        for i in sample["sid"]:
            popularity[i] = popularity.get(i,0) + 1
    popularity_tensor = torch.zeros(num_items + 1, dtype=torch.float16)
    
    for key in popularity:
        popularity_tensor[key] = popularity[key]
    
    return popularity_tensor





# input: data DictSequentialDataset
# node index: check different for users and items
# u_t = torch.LongTensor(train_df.user_id_idx)
# i_t = torch.LongTensor(train_df.item_id_idx) + n_users

# #I create the edge index by stacking the two tensors
# train_edge_index = torch.stack((
#   torch.cat([u_t, i_t]),
#   torch.cat([i_t, u_t])
# )).to(device)



# {'uid': tensor(1),
#  'in_sid': tensor([   0,    0, 2452, 1286,  770]),
#  'out_sid': tensor([[   0],
#          [   0],
#          [1286],
#          [ 770],
#          [ 955]]),
#  'in_timestamp': tensor([        0,         0, 978300019, 978300055, 978300055]),
#  'out_timestamp': tensor([[        0],
#          [        0],
#          [978300055],
#          [978300055],
#          [978300055]])}



'''
def load_item_info(dataset_raw_folder, dataset_name):   #Is it really needed?
    ML_genres = np.array(["Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"])

    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder,'u.item')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ["movie id", "movie title", "release date", "video release date",
              "IMDb URL", "unknown", *ML_genres]
        df["movie title"] = make_title_better(df["movie title"])
    elif dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder,'movies.dat')
        df = pd.read_csv(file_path, sep='::', header=None, encoding='latin-1')
        vecs = []
        for _,row in df.iterrows():
            row_genres = row.iloc[-1].split("|")
            vec = np.zeros(len(ML_genres)).astype(int)
            for genre in row_genres:
                vec[np.where(ML_genres==genre)[0]] = 1
            vecs.append(vec)
        df = df.join(pd.DataFrame(vecs),rsuffix="Cat")
        df.drop("2",axis=1,inplace=True)
        df.columns = ["movie id", "movie title", *ML_genres]
        df["movie title"] = make_title_better(df["movie title"])
    else:
        raise NotImplementedError
    return df

def make_title_better(lst): #Is it really needed?
    new_titles = []
    for complete_title in lst:
        #if "," in complete_title:
            #print(complete_title)
        year = complete_title.split("(")
        title = "(".join(year[:-1]).strip()
        year = year[-1][:-1]
        if "(" in title:
            title2 = title.split("(")
            title = "(".join(title2[:-1]).strip()
            title2 = title2[-1][:-1]
            titles = [title,title2]
        else:
            titles = [title]
        
        articles = ["A","An","The","Il","L'","La","Le","Les","O","Das","Der","Die","Det"]
        for i,tlt in enumerate(titles):
            if len(tlt)!=0:
                for article in articles:
                    app = 2+len(article)
                    if tlt[-app:] == ", "+article:
                        if "'" in article: app-=1
                        titles[i] = article+" "+tlt[:-app]
                #if "," in complete_title and tlt==titles[i]: print(tlt,titles[i])
        title = titles[0]
        for tlt in [*titles[1:],year]:
            title += " ("+tlt+")"
        #if "," in complete_title: print(title)
        new_titles.append(title)
    return new_titles

def load_genre_info(dataset_raw_folder, dataset_name):  #Is it really needed?
    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder,'u.genre')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ['genre', 'gid']
    elif dataset_name == "ml-1m":
        df = load_genre_info(os.path.join(*dataset_raw_folder.split("/")[:-1],"ml-100k"),"ml-100k")
        df = df.iloc[1:]
        df["gid"] = df["gid"]-1
        df.reset_index(drop=True, inplace=True)
    else:
        raise NotImplementedError
    return df

def load_user_info(dataset_raw_folder, dataset_name): #Is it really needed?
    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder,'u.user')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ["user id", "age", "gender", "occupation", "zip code"]
        df.replace({"gender":{"M":"Male","F":"Female"}}, inplace=True)
    elif dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder,'users.dat')
        df = pd.read_csv(file_path, sep='::', header=None, encoding='latin-1')
        df.columns = ["user id", "gender", "age", "occupation", "zip code"]
        df.replace({"age":{1: "Under 18",
                            18: "18-24",
                            25: "25-34",
                            35: "35-44",
                            45: "45-49",
                            50: "50-55",
                            56: "56+"}}, inplace=True)
        df.replace({"gender":{"M":"Male","F":"Female"}}, inplace=True)
        df.replace({"occupation":{0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student", 5: "customer service",
6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist",
16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}},inplace=True)
    else:
        raise NotImplementedError
    return df
'''


import scipy.sparse as sp
def get_graph_representation(list_of_lists):
    num_users = len(list_of_lists)
    num_items = np.max(np.concatenate([[y for y in x] for x in list_of_lists]))
    num = num_users+1+num_items+1
    matrix = sp.lil_matrix((num, num), dtype=np.float32)
    for user_id, items_list in enumerate(list_of_lists):
        for item_id in items_list:
            a = user_id
            b = item_id+(num_users+1)
            matrix[a, b] = 1
            matrix[b, a] = 1

    rowsum = np.array(matrix.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(matrix)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    coo = norm_adj.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    graph = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
    
    return graph