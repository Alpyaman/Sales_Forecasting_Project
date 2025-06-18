import pandas as pd

def load_and_merge_data():
    train = pd.read_csv('data/train.csv')
    stores = pd.read_csv('data/stores.csv')
    features = pd.read_csv('data/features.csv')

    # Converting date to datetime
    train['Date'] = pd.to_datetime(train['Date'])
    features['Date'] = pd.to_datetime(features['Date'])

    # Merging datasets
    df = pd.merge(train, stores, on='Store', how='left')
    df = pd.merge(df, features, on=['Store', 'Date'], how='left')

    # Fill missing values
    df.fillna(0, inplace=True)

    # Sorting by store, department and date
    df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

    return df


def add_features(df):
    # Data Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

    # Sorting by Store, Department, Date for lagging
    df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

    # Lagging features - 1 week and 4 weeks ago
    df['Lag_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    df['Lag_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)

    # Rolling mean and std for last 4 weeks
    df['Rolling_Mean_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=4).mean())
    df['Rolling_Std_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=4).std())

    # Filling missing lag/rolling values with 0
    df[['Lag_1', 'Lag_4', 'Rolling_Mean_4', 'Rolling_Std_4']] = df[['Lag_1', 'Lag_4', 'Rolling_Mean_4', 'Rolling_Std_4']].fillna(0)

    # Store type encoding (one-hot)
    store_type_dummies = pd.get_dummies(df['Type'], prefix='StoreType')
    df = pd.concat([df, store_type_dummies], axis=1)

    return df
