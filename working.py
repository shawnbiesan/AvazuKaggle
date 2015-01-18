import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction import FeatureHasher
import zipfile
import numpy as np
import scipy as sp


def try_get_important_features(num_batches):
    data_it = pd.read_csv('train.csv', chunksize=40428)
    imp_features = None
    model = SGDClassifier(loss='log', warm_start=True, penalty='l1')
    fh = FeatureHasher(input_type='string')
    for i, chunk in enumerate(data_it):
        if i >= num_batches:
            return list(imp_features)

        feature_matrices = []
        Y = chunk.click
        for feature in columns_to_dummy:
            col = chunk[feature]
            feature_matrices.append(fh.transform(col.apply(str).values))
        X = sp.sparse.hstack(feature_matrices)
        model.partial_fit(X, Y, classes=[0, 1])
        if i == 0:
            imp_features = np.flatnonzero(model.coef_)
        else:
            imp_features = set(np.flatnonzero(model.coef_)) & set(imp_features)



columns_to_dummy = ['C1', 'site_id', 'site_domain', 'site_category',
                    'app_id', 'app_domain', 'app_category','C14',
                    'C15','C16','C17','C18','C19','C20',
                    'device_id',
                    #'device_ip',
                    'device_type',
                    'device_conn_type',
                    ]

sample = pd.read_csv('sampleSubmission.csv')


train = pd.read_csv('train.csv', chunksize=40428)
test = pd.read_csv('test.csv', chunksize=45774)
num_batches = 100

m = try_get_important_features(10)

model = SGDClassifier(loss='log', warm_start=True, penalty='l2')
fh = FeatureHasher(input_type='string')
for i, chunk in enumerate(train):
    #if i >= num_batches:
    #    break
    feature_matrices = []
    print i
    Y = chunk.click
    for feature in columns_to_dummy:
        col = chunk[feature]
        feature_matrices.append(fh.transform(col.apply(str).values))
    #X = sp.sparse.hstack(feature_matrices, format='dok')[:, m]
    X = sp.sparse.hstack(feature_matrices)
    model.partial_fit(X, Y, classes=[0, 1])

for i, chunk in enumerate(test):
    feature_matrices = []
    for feature in columns_to_dummy:
        col = chunk[feature]
        feature_matrices.append(fh.transform(col.apply(str).values))
    #X = sp.sparse.hstack(feature_matrices, format='dok')[:, m]
    X = sp.sparse.hstack(feature_matrices)
    result = model.predict_proba(X)
    result = result[:, 1]  # just the prob of 1
    sample['click'][i*test.chunksize: (i+1)*test.chunksize] = result
    print i

sample.to_csv('resultmixed.csv', index=False)

zipper = zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED)
print 'zipping results'
zipper.write('resultmixed.csv')

print model.coef_


