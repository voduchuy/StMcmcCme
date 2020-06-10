import numpy as np
from scipy.io import loadmat

def load_data():
    cond = 'NoInhibitors'
    ispec = 0

    # 1st row IL1beta, 2nd row TNFalpha
    data = loadmat("Data/InflamResponseData.mat")
    obsr_tags = ['0hr', '30mins', '1hr', '2hr', '4hr']
    data_snapshots = []
    for i in range(0, len(obsr_tags)):
        X = data['outputs_' + obsr_tags[i] + '_' + cond]
        X = np.ascontiguousarray(X[ispec, :])
        data_snapshots.append(X)
    return data_snapshots

data_snapshots = load_data()
np.savez('il1b_data.npz', mrna_counts=data_snapshots)