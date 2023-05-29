import sys
import datetime
import requests
import os

def get_now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_log_message(message):
    print(get_now() + ' - ' + message)
    sys.stdout.flush()

def hasConnection():
    try:
        request = requests.get("www.google.com", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        add_log_message("Sin conexión a internet. Guardamos ficheros en local.")
    else:
        add_log_message("Con conexión a internet. Subimos ficheros a Azure.")

def getOldFile(path):
    path = os.path.join(os.getcwd(),"pending",path)
    files = os.listdir(path)
    old_file = None
    old_time = float('inf')

    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            mod_time = os.path.getmtime(file_path)
            if file[0] != '_' and mod_time < old_time:
                old_time = mod_time
                old_file = file

    if old_file is not None:
        add_log_message("Fichero mas antiguo "+ old_file)
        return path,file
    else:
        add_log_message("No se encuentran ficheros en "+ path)
        return False, False
    
def renameFile(path, filename, newname):
    current_path = os.path.join(path, filename)
    new_path = os.path.join(path, newname)
    if os.path.isfile(current_path):
        os.rename(current_path, new_path)
        add_log_message("El archivo se ha renombrado correctamente de "+current_path + " a "+new_path)
        return newname
    else:
        add_log_message("No se puede renombrar el archivo" +current_path+" porque no existe.") 
        return False
    
def deleteFile(path, filename):
    file_path = os.path.join(path, filename)
    try:
        os.remove(file_path)
        add_log_message("Archivo eliminado correctamente: "+ str(file_path))
    except OSError as e:
        add_log_message("Error al eliminar el archivo: " + str(e))