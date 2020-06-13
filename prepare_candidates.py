import geopandas as gpd
import progressbar
import random
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np



gp = gpd.read_file('dataset/Shapes/GP_Informal_settlement2017.shp')

lats = []
lons = []
labels = []

for i in progressbar.progressbar(range(1000000)):
    #lon = 28 + random.random()*0.5
    lon = random.randint(27,28)+random.random()*1.0
    #lat = -26 + random.random()*0.5
    lat = -(random.randint(25,26)+random.random()*1.0)
    p = Point(lon, lat)
    IN = False
    for geom in gp.geometry:
        if p.within(geom):
            IN = True
    lats.append(lat)
    lons.append(lon)
    labels.append(IN)
ls = pd.DataFrame({
    'LAT':lats,
    'LON':lons,
    'Label':labels
})
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(ls)
(unique, counts) = np.unique(ls["Label"].values, return_counts=True)
print(unique, counts)
#ls.sample(10000)

ls.to_csv('dataset/dataset_shapes.csv', index=False)
