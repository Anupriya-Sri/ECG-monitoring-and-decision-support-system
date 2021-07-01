from read_write import Loader, Writer
from preprocessing import Extractor
import pandas as pd
import numpy as np
import pickle

def get_started():

    path = r"D:\Sabudh Work\New folder\machine_learning_model\Training_Set\\"
    labels = pd.read_csv(r"D:\Sabudh Work\New folder\machine_learning_model\train_labels.csv")
    fs = 500
    loader_obj = Loader(path)
    extractor_obj = Extractor(fs)
    batches = loader_obj.load_batch(labels.Recording)
    numerical_features = []
    wavelet_features = []
    hos_features = []
    lbp_features = []
    print('Extracting features, might take a while...')

    idx = -1
    for batch_data in batches:
        for rec in batch_data:
            idx += 1
            try:
                features = extractor_obj.get_features(rec)
                numerical_features.append(features[:8])
                wavelet_features.append(features[8])
                lbp_features.append(features[9])
                hos_features.append(features[10])
            except:
                print("Exception")
                print(rec.shape)
                print(idx)
                print("\n")


    X = np.hstack(tuple(map(np.array, [numerical_features, wavelet_features, hos_features, lbp_features])))

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(X, f)

    print('Done extracting.')

if __name__ == '__main__':
    get_started()
