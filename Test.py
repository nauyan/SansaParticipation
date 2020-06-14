import keras
import glob
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os 
import pandas as pd
import json
import progressbar
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


testList = glob.glob("dataset/test_set/"+"*jpeg")

count = 0
Images = []
Names = []
for x in progressbar.progressbar(testList):
    name = x
    img = img_to_array(load_img(name, color_mode='grayscale'))
    img = img/255.0
    Images.append(img)
    Names.append(name)
    #if count == 10:
       #break
    count = count + 1


Images = np.asarray(Images)
Names = np.asarray(Names)

import efficientnet.keras

model = keras.models.load_model("Final.h5") 

pred = model.predict(Images, verbose=1)
print(pred)

Names_Sub = []
for val in Names:
    tmp = os.path.splitext(val)[0]
    Names_Sub.append(tmp[17:])
    
df = pd.DataFrame()
df['ID']  = Names_Sub
df['Label']  = pred
#print(df)

import os
os.remove("submission.csv")
print("File Removed!")
df.to_csv('submission.csv', index=False)