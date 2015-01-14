import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction import FeatureHasher
import zipfile
import scipy as sp


columns_to_dummy = ['C1', 'site_id', 'site_domain', 'C14', 'C15','C16','C17','C18','C19','C20']
sample = pd.read_csv('sampleSubmission.csv')


train = pd.read_csv('train.csv', chunksize=40428)
test = pd.read_csv('test.csv', chunksize=45774)
model = SGDClassifier(loss='log', warm_start=True)
fh = FeatureHasher(input_type='string')
for chunk in train:
    feature_matrices = []
    print 'chunk'
    Y = chunk.click
    for feature in columns_to_dummy:
        col = chunk[feature]
        feature_matrices.append(fh.transform(col.apply(str).values))
    X = sp.sparse.hstack(feature_matrices)
    model.partial_fit(X, Y, classes=[0, 1])

for i, chunk in enumerate(test):
    feature_matrices = []
    for feature in columns_to_dummy:
        col = chunk[feature]
        feature_matrices.append(fh.transform(col.apply(str).values))
    X = sp.sparse.hstack(feature_matrices)
    result = model.predict_proba(X)
    result = result[:, 1]  # just the prob of 1
    sample['click'][i*test.chunksize: (i+1)*test.chunksize] = result
    print i

sample.to_csv('resultmixed.csv', index=False)

zipper = zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED)
print 'zipping results'
zipper.write('resultmixed.csv')




