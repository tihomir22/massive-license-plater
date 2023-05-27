from azure.storage.blob import BlobServiceClient
import pyarrow.parquet as pq
import os
from ultralytics import YOLO
from predictFn import PredictLicensePlate
import cv2
from io import BytesIO
import base64
import numpy as np
import argparse

#print("\n INIT CHILD \n")
# Configurar el nivel de registro a un nivel alto que descarte todos los registros

connectionString =  os.getenv('AZURE_STRING_CONNECTION')
blob_service_client = BlobServiceClient.from_connection_string(connectionString)
container_name = "cars"
container_client = blob_service_client.get_container_client(container_name)

dirnameYolo=os.path.abspath("yoloBest.pt")
model = YOLO(os.path.join(dirnameYolo))
instance = PredictLicensePlate(model)

parser = argparse.ArgumentParser(description='Script para eliminar la foto de un dataframe.')
parser.add_argument("--blob_name", help="Añade el blob_name. Ejemplo: cars_00020610-7d8b-40e9-867d-d53559a4e767.parquet.", default="streets")
args = parser.parse_args()

blob_name = args.blob_name

blob_client = container_client.get_blob_client(blob_name)
stream = BytesIO()
blob_client.download_blob().download_to_stream(stream)
stream.seek(0)

parquet_file = pq.ParquetFile(stream)
df = parquet_file.read().to_pandas()
list_license_plates = []
#print("processing new child")
#tmp
#df = df.head(10)
for index, row in df.iterrows():
    image_b64 = row['photo']
    filestr = image_b64.split(',')[1]
    img_bytes = base64.b64decode(filestr)
    img_stream = BytesIO(img_bytes)
    imgArray = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    res = instance.doPredict(imgArray)
    list_license_plates.append(res)

df['matricula'] = list_license_plates
df = df.drop('photo', axis=1)

df.to_csv("results/"+blob_name+'.csv')