import os
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
import progressbar


tif_files = ["aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2527A.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2527B.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2527C.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2527D.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2528A.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2528B.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2528C.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2528D.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2627A.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2627B.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2627C.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2627D.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2628A.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2628B.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2628C.tif dataset/Image.tif",
             "aws s3 cp s3://eohackathon-covid19/Hackthon_Data/Gauteng/2628D.tif dataset/Image.tif"]

images_names = pd.read_csv('dataset/dataset_shapes_refined.csv')
             
for name in tif_files:
    print(name)
    os.system(name)
    dataset = rasterio.open('dataset/Image.tif')
    
    # Load bands into RAM
    red, green, blue = dataset.read(1), dataset.read(2), dataset.read(3)

    # Let's do 5 points for now - use df.values to do this for the whole dataset
    #for ID, lat, lon, label in df.sample(5).values:
    count = 0 
    #for lat, lon, label in dataset.values:
    for ind in progressbar.progressbar(images_names.index):
    
        lat = images_names['LAT'][ind]
        lon = images_names['LON'][ind]
        label = images_names['Label'][ind]
    
    
        # Blank image
        im = np.zeros((384,384,3), np.uint8)
    
        # Get pixel coords
        row, col = dataset.index(lon, lat)
        try:
            # Add image data
            for i, band in enumerate([red, green, blue]):
                im[:,:,i] = band[row-192:row+192, col-192:col+192]
            images_names = images_names.drop([ind])
        except:
            continue
        # Save with the location in the name
        im = Image.fromarray(im)
        pth = 'dataset/train_set/0/'
        if label == True:
            pth = 'dataset/train_set/1/'
        im.save(pth+f"im_{count}_{lat}_{lon}.jpeg")
        count = count + 1
    print(count)