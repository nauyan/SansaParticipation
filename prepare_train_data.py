import os
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
import progressbar


train = pd.read_csv('dataset/Train.csv')
#print(test)

train_file = "aws s3 cp 's3://eohackathon-covid19/Hackthon_Data/Gauteng/2528C.tif' dataset/Image.tif"
os.system(train_file)

dataset = rasterio.open('dataset/Image.tif')

# Load bands into RAM
red, green, blue = dataset.read(1), dataset.read(2), dataset.read(3)

# Let's do 5 points for now - use train.values to do this for the whole dataset
#for ID, lat, lon, label in train.sample(5).values:
count = 0 
for ID, lat, lon, label in progressbar.progressbar(train.values):
    
    # Blank image
    im = np.zeros((384,384,3), np.uint8)
    
    # Get pixel coords
    row, col = dataset.index(lon, lat)
    
    # Add image data
    for i, band in enumerate([red, green, blue]):
        im[:,:,i] = band[row-192:row+192, col-192:col+192]
    
    # Save with the location in the name
    im = Image.fromarray(im)
    pth = 'dataset/train_set/0/'
    if label == True:
        pth = 'dataset/train_set/1/'
    im.save(pth+f"im_{ID}_{lat}_{lon}.jpeg")
    count = count + 1
print(count)
