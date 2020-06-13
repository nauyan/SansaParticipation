import pandas as pd

ls = pd.read_csv('dataset/dataset_shapes.csv')
# Keeping 5% of the negatives (feel free to tweak to taste!):
l_sample = pd.concat([ls.loc[ls.Label==True], ls.loc[ls.Label==False].sample(frac=0.002)])
#l_sample.plot(kind='scatter', x='LON', y='LAT', c=l_sample['Label'].map(lambda x: (1, 0, 0.5) if x else (0, 0, 0)))

print(l_sample.Label.sum(), l_sample.shape, l_sample.Label.sum()/l_sample.shape[0])
l_sample.to_csv('dataset/dataset_shapes_refined.csv', index=False)