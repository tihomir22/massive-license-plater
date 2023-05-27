from azure.storage.blob import BlobServiceClient
import pyarrow.parquet as pq
import os
from dotenv import load_dotenv
import psutil
import time
import subprocess
import platform

load_dotenv()

connectionString = os.getenv('AZURE_STRING_CONNECTION')
blob_service_client = BlobServiceClient.from_connection_string(
    connectionString)
container_name = "cars"
container_client = blob_service_client.get_container_client(container_name)
blobs = container_client.list_blobs()

MAX_LIMIT = 80
WAIT_TIME = 15

if platform.system() == 'Windows':
    python = "python"
else:
    python = "python3"


def resourcesExceeded():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    print("Uso de CPU: {}%".format(cpu_percent))
    print("Uso de memoria: {}%".format(memory_percent))
    return cpu_percent >= MAX_LIMIT or memory_percent >= MAX_LIMIT

total_blobs = 0
for blob in blobs:
    total_blobs += 1
print("Hay "+str(total_blobs) +" blobs que procesar ")
blobs = container_client.list_blobs()
cont = 0
for blob in blobs:

    if not blob.name.endswith('.parquet'):
        continue

    while True:
        if resourcesExceeded():
            time.sleep(WAIT_TIME)
            continue
        else:
            subprocess.Popen(
                [python, "cleanSlave.py", "--blob_name", str(blob.name)])
            break
        
    print("\n \n")
    print("Actualmente se ha lanzado el thread numero "+str(cont))
    print("En total son "+str(total_blobs))
    print("\n \n")
    
    cont = cont + 1
    if cont == 5:
        break
