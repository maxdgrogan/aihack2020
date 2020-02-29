import torch

def create_inout_sequences(input_data, tw):
    X = []
    Y = []
    L = len(input_data)
    for i in range(L-tw):
        seq = input_data[i:i+tw]
        label = input_data[i+tw:i+tw+1]
        X.append(seq)
        Y.append(label)

    return torch.stack(X), torch.stack(Y)


def get_loaders(train_data, test_data, args):
    train_X, train_Y = create_inout_sequences(train_data, args.train_window)
    test_X, test_Y = create_inout_sequences(test_data, args.train_window)


    train_dataset = torch.utils.data.TensorDataset(train_X,train_Y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size,shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_X,test_Y)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.batch_size,shuffle=True)
    
    return train_loader, test_loader
