import pandas as pd

def loc(df, column, condition, value):
    if condition == 'eq':
        return df.loc[df[column] == value]
    elif condition == 'in':
        return df.loc[df[column].isin(value)]
    elif condition == 'not in':
        return df.loc[~df[column].isin(value)]
