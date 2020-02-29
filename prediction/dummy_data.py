import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch

def get_data(test_portion = 0.2):
    flight_data = sns.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)

    assert test_portion>=0 and test_portion<1

    total_size = len(all_data)
    train_size = int((1-test_portion) * total_size)

    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
    test_data_normalized = scaler.fit_transform(test_data .reshape(-1, 1))

    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

    return train_data_normalized, test_data_normalized