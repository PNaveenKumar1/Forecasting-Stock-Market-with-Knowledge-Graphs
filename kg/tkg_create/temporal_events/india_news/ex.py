import os
import pickle
import pandas as pd

with open("newsdata_nifty.pkl", "rb") as f:
    data = pickle.load(f)

counter = 0
for ticker, news in data.items():
    for event in news:
        counter += 1
print(counter)