from azure.storage.blob import BlobServiceClient
import pyarrow.parquet as pq

import os
from ultralytics import YOLO
import yaml
from predictFn import PredictLicensePlate
import re
import cv2
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import imutils
from PIL import Image

dirnameYolo=os.path.abspath("yoloBest.pt")
model = YOLO(os.path.join(dirnameYolo))
instance = PredictLicensePlate(model)

connectionString = "DefaultEndpointsProtocol=https;AccountName=iessimarroes01;AccountKey=E5B9vCkp4No1NdRh7xo/iBL5xEWQnEWXqAAzlkY8ZNPrVQGbcvcjH34Vs+OSdN9d1+/USTeZgTjQ+AStdd9j3w==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connectionString)

container_name = "cars"
container_client = blob_service_client.get_container_client(container_name)

blobs = container_client.list_blobs()

for blob in blobs:
    print(blob.name)
    if not blob.name.endswith('.parquet'):
        continue

    blob_client = container_client.get_blob_client(blob.name)
    stream = BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    stream.seek(0)

    parquet_file = pq.ParquetFile(stream)
    df = parquet_file.read().to_pandas()

    for index, row in df.iterrows():
        image_b64 = row['photo']
        filestr = image_b64.split(',')[1]
        img_bytes = base64.b64decode(filestr)
        img_stream = BytesIO(img_bytes)
        imgArray = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
        #print(imgArray[0])
        print(image_b64)
        row['matricula'] = instance.doPredict(imgArray)
        row = row.drop('photo')

        df.loc[index] = row

        #print(row)
        #break
    
    print(df)
    
    break