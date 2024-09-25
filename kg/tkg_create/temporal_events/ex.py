import pickle
import json
import pandas as pd

with open("news_headlines_cleaned.pkl", "rb") as f:
    headlines_json = pickle.load(f)

news_id_to_news = {}
ii = 0
for ticker, v in headlines_json.items():
    for news in v:
        if news[0] in news_id_to_news:
            news_id_to_news[news[0]][1].append(ticker)
        else:
            ii += 1
            news_id_to_news[news[0]] = [[news[2], news[1]], [ticker]]

print(ii, len(news_id_to_news))