import time
import argparse
import common
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='Script que recorre directorio y envia los mensajes a Azure.')
parser.add_argument("--container_name", help="Añade el nombre del contenedor. Ejemplo: streets.", default="streets")
args = parser.parse_args()

container_name = args.container_name

load_dotenv()

connection_string = os.getenv('AZURE_STRING_CONNECTION')

while True:
    path,file = common.getOldFile(container_name)
    if(path and file):
        tmp_name = "_"+file
        tmp_file = common.renameFile(path,file,tmp_name)
        if(tmp_file):
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client(container_name)
                with open(os.path.join(path, tmp_file), "rb") as _file:
                    container_client.upload_blob(name=file, data=_file)
            except Exception as e:
                common.renameFile(path,tmp_name,file)
                common.add_log_message("Error al subir el archivo al contenedor: "+ str(e))
                common.add_log_message(str(_file))
            else:
                common.deleteFile(path,tmp_file)
                common.add_log_message("Archivo subido exitosamente al contenedor: " + container_name + "con nombre de blob:" + file)
        else:
            common.add_log_message("No se puede renombrar el fichero")
            
    time.sleep(1)