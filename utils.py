import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import torch 
from torch_geometric.data import Data

# ------- H1: DATASET UTILS -------------

# ------- H2: Preprocess Data -------------
def window_scale_divison(df, W, T, company_to_id, ticker):
    """
        Returns the window of input and target values scaled by dividing with the
        last value of the previous window.

        Problems: With large W and T
    """
    SMOOTH = 0.00001
    list_df = [(
                    (df['Open'][i+1:i+W+1] / df['Open'][i:i+1].values+SMOOTH).values, 
                    (df['High'][i+1:i+W+1] / df['High'][i:i+1].values+SMOOTH).values,           
                    (df['Low'][i+1:i+W+1] / df['Low'][i:i+1].values+SMOOTH).values, 
                    (df['Close'][i+1:i+W+1] / df['Close'][i:i+1].values+SMOOTH).values,           
                    (df['Volume'][i+1:i+W+1] / (df['Volume'][i:i+1].values+SMOOTH)).values, 
                    company_to_id[ticker],  
                    df[i+1:i+W+1]['Date'], 
                    [
                        (df['Close'][i+W+T:i+W+T+1] / df['Close'][i+W:i+W+1].values).values
                        for T in [1, 5, 20]
                    ], 
                    df['Close'][i+W:i+W+1],
                    df.iloc[i+W:i+W+1, 7:].values,
                    [
                        (df['Low'][i+W+T:i+W+T+1]).values
                        for T in [0, 1, 5, 20]
                    ], 
                    [
                        (df['High'][i+W+T:i+W+T+1]).values
                        for T in [0, 1, 5, 20]
                    ]
                ) 
                for i in range(df.shape[0]-W-T)
            ]
    return list_df

# ------- H2: Create Data -------------
def create_batch_dataset(INDEX, W, T=20, problem='value', fast = False):

    directory = "data/" + "nasdaq_2" + "/"

    company_to_id = {}
    company_id    = 0

    dataset = []
    df_map = {}
    skipped_ticker = []
    total = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(filename)
            ticker, name = filename.split("-")
            df = pd.read_csv(f)

            # df = df.dropna()
            df = df.fillna(method='ffill')

            if df.shape[0] <= 2800:    # 13 years
                print("Skipping file: Less Training-Testing samples [{0} samples]".format(df.shape[0]))
                skipped_ticker.append(ticker)
                continue
            total += 1

            if ticker not in company_to_id:
                company_to_id[ticker] = company_id
                company_id += 1
            
            
            if df.shape[0] > 2800:
                df = df.iloc[-2800:]
        
            """
            annual_df = pd.read_csv("kg/fundamentals/macrotrends/combined-preprocessed/"+ticker+"-annual.csv")
            quarterly_df = pd.read_csv("kg/fundamentals/macrotrends/combined-preprocessed/"+ticker+"-quarterly.csv")

            df['Date'] = pd.to_datetime(df['Date'])
            annual_df['Date'] = pd.to_datetime(annual_df['Date'])
            quarterly_df['Date'] = pd.to_datetime(quarterly_df['Date'])

            # sort annual_df based on date
            annual_df = annual_df.sort_values('Date')
            quarterly_df = quarterly_df.sort_values('Date')
            

            # Join df on date with value greater than given
            df = pd.merge_asof(df, annual_df, on='Date', direction='backward')
            df = pd.merge_asof(df, quarterly_df, on='Date', direction='backward')

            # Fill NaN values with previous values
            df = df.fillna(method='ffill')
            print(df.shape)
            """

            list_df = [(
                    df.iloc[i+W:i+W+1, 7:].shape
                ) 
                for i in range(df.shape[0]-W-T)
            ]

            list_df = window_scale_divison(df, W, T, company_to_id, ticker)

            df_map[company_to_id[ticker]] = list_df
    
    kg_file_name = './kg/tkg_create/temporal_kg.pkl'
    if INDEX == 'nifty500':
        kg_file_name = './kg/tkg_create/temporal_kg_nifty.pkl'
    with open(kg_file_name, 'rb') as f:
        pkl_file = pickle.load(f)
        relation_kg = pkl_file['temporal_kg']

    for i in range(len(list_df)):
        cur_data = []
        start_time, end_time = list_df[i][6].iloc[-1], list_df[i][6].iloc[-1] 
        start_time = pd.to_datetime(start_time, utc=True)
        end_time = pd.to_datetime(end_time, utc=True) + pd.offsets.Day(1)
        start_time.tz_localize(None)
        end_time.tz_localize(None)
        relation_kg['expiry_ts'] = pd.to_datetime(relation_kg['expiry_ts'], utc=True)
        relation_kg['timestamp'] = pd.to_datetime(relation_kg['timestamp'], utc=True)
        non_temporal_time = pd.to_datetime('1970-01-01', utc=True)
        mask = (relation_kg['timestamp'] >= start_time) & (relation_kg['timestamp'] < end_time) & (relation_kg['expiry_ts'] >= end_time) | (relation_kg['timestamp'] == non_temporal_time)
        tkg = relation_kg.loc[mask]

        head, relation, tail = torch.Tensor([int(x) for x in tkg['head'].values]).long(), torch.Tensor([int(x) for x in tkg['relation'].values]).long(), torch.Tensor([int(x) for x in tkg['tail'].values]).long()
        #head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        #print(start_time, tkg)

        start_ts_years = torch.Tensor(tkg['timestamp'].dt.year.values).long() - 2000
        start_ts_years[start_ts_years <= 0] = 0
        start_ts_months = torch.Tensor(tkg['timestamp'].dt.month.values).long()
        start_ts_months[start_ts_years <= 0] = 0
        start_ts_days = torch.Tensor(tkg['timestamp'].dt.day.values).long()
        start_ts_days[start_ts_years <= 0] = 0
        start_ts_hours = torch.Tensor(tkg['timestamp'].dt.hour.values).long()
        start_ts_hours[start_ts_years <= 0] = 0
        start_ts_minutes = torch.Tensor(tkg['timestamp'].dt.minute.values).long()
        start_ts_minutes[start_ts_years <= 0] = 0
        start_ts_seconds = torch.Tensor(tkg['timestamp'].dt.second.values).long()
        start_ts_seconds[start_ts_years <= 0] = 0

        ts = torch.stack([start_ts_years, start_ts_months, start_ts_days, start_ts_hours, start_ts_minutes, start_ts_seconds], dim=1)
        
        #print(tkg['timestamp'], ts, ts.shape)
        temporal_kg = (head, relation, tail, ts)

        for j in range(company_id):
            cur_data.append(df_map[j][i])

        dataset.append((cur_data, temporal_kg)) 

    print("Skipped Tickers: ", skipped_ticker)

    sector_graph = open("kg/sector/sector_hypergraph_"+INDEX+".txt", "r").readlines()

    # Unidirectional Homogeneous Graph
    sector_map = {}
    graph_dataset = []
    for lines in sector_graph[1:]:
        lines = lines[:-1]
        tickers = lines.split("\t")[2:]
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if tickers[i] not in company_to_id or tickers[j] not in company_to_id:
                    continue
                if tickers[i] + "-" + tickers[j] not in sector_map:
                    graph_dataset.append([company_to_id[tickers[i]], company_to_id[tickers[j]]])
                    graph_dataset.append([company_to_id[tickers[j]], company_to_id[tickers[i]]])
                    sector_map[tickers[i] + "-" + tickers[j]] = 1
                    sector_map[tickers[j] + "-" + tickers[i]] = 1

    edge_index = torch.Tensor(graph_dataset).t().contiguous()
    x = torch.randn(len(company_to_id.items()), 8)

    print("Number of edges in pairwise graph: ", len(graph_dataset))

    graph_data = Data(x=x, edge_index=edge_index.long())

    hyperedge_index, hyper_x = get_sector_hypergraph(company_to_id)
    hyper_data = {
        'hyperedge_index': hyperedge_index,
        'x': hyper_x
    }
    
    return dataset, company_to_id, graph_data, hyper_data

def load_dataset_graph(save_path):
    with open(save_path, 'rb') as handle:
        b = pickle.load(handle)

    dataset = b['train']
    company_to_id = b['company']
    graph = b['graph']
    hyper_data = b['hyper_graph']

    return dataset, company_to_id, graph, hyper_data

def save_dataset_graph(save_path, dataset, company_to_id, graph, hyper_data):
    print("--- Saving Dataset ---")
    save_data = {'train': dataset, 'company': company_to_id, 'graph': graph,
                    'hyper_graph': hyper_data}

    with open(save_path, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_or_create_dataset_graph(INDEX, W, T, save_path, problem, fast):
    if fast == True:
        print("--- Creating Dataset ---")
        dataset, company_to_id, graph, hyper_data = create_batch_dataset(INDEX, W, T, problem, fast)
    elif os.path.isfile(save_path):
        print("--- File exists: Loading Dataset ---")
        dataset, company_to_id, graph, hyper_data = load_dataset_graph(save_path)
    else:
        print("--- Creating Dataset ---")
        dataset, company_to_id, graph, hyper_data = create_batch_dataset(INDEX, W, T, problem, fast)
        save_dataset_graph(save_path, dataset, company_to_id, graph, hyper_data)

    return dataset, company_to_id, graph, hyper_data
    
    
# METRICS UTILS

def mean_absolute_percentage_error(y_true, y_pred): 
    return (((y_true - y_pred) / y_true).abs()).mean() * 100
    #return ((y_true - y_pred).abs()).mean() * 100

def root_mean_square_error(y_true, y_pred, scale = None): 
    if scale == None:
        return ((y_true - y_pred) ** 2).mean() ** (1/2)
    else:
        return (((y_true - y_pred)*scale) ** 2).mean() ** (1/2)

def mean_square_error(y_true, y_pred, scale = None):
    if scale == None:
        return ((y_true - y_pred) ** 2).mean() 
    else:
        return (((y_true - y_pred)*scale.unsqueeze(dim=1)) ** 2).mean() 

# 3. KG Loader

def get_sector_hypergraph(company_to_id):
    # HyperGraph
    sector_graph = open("kg/sector/sector_hypergraph_nasdaq100.txt", "r").readlines()

    n = len(company_to_id.items())
    hyperedge_index = [[], []]
    edge = 0
    for lines in sector_graph[1:]:
        lines = lines[:-1]
        tickers = lines.split("\t")[2:]
        for i in range(len(tickers)):
            if tickers[i] not in company_to_id:
                continue
            hyperedge_index[0].extend([company_to_id[tickers[i]]])
            hyperedge_index[1].extend([edge])
        edge += 1

    hyperedge_index = torch.Tensor(hyperedge_index).long()
    hyper_x = torch.randn(len(company_to_id.items()), 8)

    return hyperedge_index, hyper_x

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--index', type=str, default="AAPL")
    #parser.add_argument('--window', type=int, default=10)
    #parser.add_argument('--test_size', type=float, default=0.2)

    #create_dataset("nasdaq100", 50, 5)
    d, s, c, g, h = create_batch_dataset("nasdaq100", 50, 5)
    print(len(d[0]), len(d[0][0]), d[0][0][0])

