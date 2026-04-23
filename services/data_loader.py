import pandas as pd

def load_data(url):
    try:
        return pd.read_csv(url)
    except:
        return None