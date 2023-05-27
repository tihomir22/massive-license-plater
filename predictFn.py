# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
######################################################################
import pytesseract
import numpy as np
import cv2
import platform
import logging
logging.basicConfig(level=logging.CRITICAL)
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

X_resize=220
Y_resize=70

import os
import re
import imutils
from skimage.transform import radon
import numpy
from PIL import Image
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft
import traceback
import pandas as pd

try:
    from parabolic import parabolic
    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax
    
class PredictLicensePlate:
    
    def __init__(self,loadedModel):
        self.model = loadedModel
        self.class_list = self.model.model.names
        
    def DetectLicenseWithYolov8 (self,img):
        TabcropLicense=[]
        results = self.model.predict(img)
        
        
        
        # imagen_pil = Image.fromarray(results[0][1])
        # imagen_pil.save("imagen2.png")
        
        
        #print(len(results[0]))
        result=results[0]
        xyxy= result.boxes.xyxy.cpu().numpy()
        confidence= result.boxes.conf.cpu().numpy()
        
        class_id= result.boxes.cls.cpu().numpy().astype(int)
        # Get Class name
        class_name = [self.class_list[x] for x in class_id]
        # Pack together for easy use
        sum_output = list(zip(class_name, confidence,xyxy))
        # Copy image, in case that we need original image for something
        out_image = img.copy()
        for run_output in sum_output :
            # Unpack
            label, con, box = run_output
            if label == "vehicle":continue
            cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            #cv2.imshow("Crop", cropLicense)
            imagen_pil = Image.fromarray(cropLicense)
            imagen_pil.save("imagen.png")
            #cv2.waitKey(0)
            TabcropLicense.append(cropLicense)

        return TabcropLicense

    def validar_matricula(self,matricula):
        regex_matricula = r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$'
        if re.match(regex_matricula, str(matricula)):
            return True
        else:
            return False
        
    def extraer_numeros(self,texto):
        numeros = ''.join(c for c in texto if c.isdigit())
        return numeros
    
    def agregar_padding(self,texto):
        texto_padding = texto.ljust(7, 'X')
        return texto_padding
        
    def reemplazar_letras_numeros(self,texto):
        diccionario_reemplazo = {
            'S': '9',
            'B': '8',
            'G': '6',
            'O': '0',
            'I': '1',
            'Z': '2',
            'A':'4',
            'T':'7',
            'E':'6'
            # Agrega más letras y sus correspondientes números
        }
        
        texto_modificado = ""
        for caracter in texto:
            if caracter in diccionario_reemplazo:
                texto_modificado += diccionario_reemplazo[caracter]
            else:
                texto_modificado += caracter
        return texto_modificado
    
    def getLicenseNumber(self,image):  
        x_off=3
        y_off=2
                
        x_resize=220
        y_resize=70
                
        Resize_xfactor=1.78
        Resize_yfactor=1.78

        BilateralOption=0
                
        TabLicensesFounded, ContLicensesFounded= self.FindLicenseNumber (image, x_off, y_off, x_resize, y_resize, \
                                    Resize_xfactor, Resize_yfactor, BilateralOption)
        contmax=0
        licensemax=""
        rows = []
       
        
        for y in range(len(TabLicensesFounded)):
            matricula = TabLicensesFounded[y]
            matricula = self.agregar_padding(matricula)
            matricula = matricula[:-2][-3:]
            matricula = self.reemplazar_letras_numeros(matricula)
           # print("Matricula A "+ str(matricula))
            rows.append({"Matricula detectada":matricula,"confianza":ContLicensesFounded[y]})
            if ContLicensesFounded[y] > contmax:
                contmax=ContLicensesFounded[y]
                licensemax=TabLicensesFounded[y]
        
        df = pd.DataFrame(rows,columns=["Matricula detectada", "confianza"])
        df = df.sort_values('confianza', ascending=False)
        filtro = df['Matricula detectada'].str.isdigit()
        df = df[filtro]
        #df = df[self.contiene_numeros(df['Matricula detectada'])]
        #matricula = licensemax
        #print("Matricula B "+ str(matricula))
        #matricula = self.agregar_padding(matricula)
        #matricula = matricula[:-2][-3:]
        #matricula = self.reemplazar_letras_numeros(matricula)
        if len(df.index) == 0:
            rows = []
            rows.append({"Matricula detectada":matricula,"confianza":0})
            df = pd.DataFrame(rows,columns=["Matricula detectada", "confianza"])
        #print(df)
        df.reset_index(drop=True, inplace=True)
        return df["Matricula detectada"][0]
    
    def contiene_numeros(self,cadena):
        return bool(re.search(r'\d', str(cadena)))

    def doPredict(self,image):
        try:
            rotationApplied = 0
            TabImgSelect =self.DetectLicenseWithYolov8(image)
            matricula = ""
            if len(TabImgSelect) > 0:
                matricula = self.getLicenseNumber(TabImgSelect[0])
               # print("First detection "+matricula)
                
            MAX_TRIES = 50
            tries = 0
            
            while rotationApplied < 360 and not matricula.isdigit():
                image=imutils.rotate(image,angle=rotationApplied)
                TabImgSelect =self.DetectLicenseWithYolov8(image)
                if len(TabImgSelect)>0:
                    matricula = self.getLicenseNumber(TabImgSelect[0])
                    #print("Rotate detection "+matricula)
               # else:
                    #print("Not detected license plate. Rotate more.")
                rotationApplied = rotationApplied + 15
                tries = tries + 1
                if tries > MAX_TRIES:
                    break
                
            if len(TabImgSelect) == 0: 
                raise ValueError("No license plate detected! Check your image.")
                
            #print("Final detection "+matricula)
            #matricula = matricula[:-2][-3:]
            #matricula = self.reemplazar_letras_numeros(matricula)
            return matricula
        except Exception as e:
           # print("Excepcion")
            traceback.print_exc()
            return False
    


    def GetRotationImage(self,image):

    
        I=image
        I = I - mean(I)  # Demean; make the brightness extend above and below zero
        
        
        # Do the radon transform and display the result
        sinogram = radon(I)
    
        
        # Find the RMS value of each row and find "busiest" rotation,
        # where the transform is lined up perfectly with the alternating dark
        # text and white lines
        
        # rms_flat does no exist in recent versions
        #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
        r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
        rotation = argmax(r)
        #print('Rotation: {:.2f} degrees'.format(90 - rotation))
        #plt.axhline(rotation, color='r')
        
        # Plot the busy row
        row = sinogram[:, rotation]
        N = len(row)
        
        # Take spectrum of busy row and find line spacing
        window = blackman(N)
        spectrum = rfft(row * window)
        
        frequency = argmax(abs(spectrum))
    
        return rotation, spectrum, frequency

    #####################################################################
    def ThresholdStable(self,image):
        # -*- coding: utf-8 -*-
        """
        Created on Fri Aug 12 21:04:48 2022
        Author: Alfonso Blanco García
        
        Looks for the threshold whose variations keep the image STABLE
        (there are only small variations with the image of the previous 
        threshold).
        Similar to the method followed in cv2.MSER
        https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
        """
    
        thresholds=[]
        Repes=[]
        Difes=[]
        
        gray=image 
        grayAnt=gray

        ContRepe=0
        threshold=0
        for i in range (255):
            
            ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
            Dife1 = grayAnt - gray1
            Dife2=np.sum(Dife1)
            if Dife2 < 0: Dife2=Dife2*-1
            Difes.append(Dife2)
            if Dife2<22000: # Case only image of license plate
            #if Dife2<60000:    
                ContRepe=ContRepe+1
                
                threshold=i
                grayAnt=gray1
                continue
            if ContRepe > 0:
                
                thresholds.append(threshold) 
                Repes.append(ContRepe)  
            ContRepe=0
            grayAnt=gray1
        thresholdMax=0
        RepesMax=0    
        for i in range(len(thresholds)):
            #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
            if Repes[i] > RepesMax:
                RepesMax=Repes[i]
                thresholdMax=thresholds[i]
                
        #print(min(Difes))
        #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
        return thresholdMax

    
    
    # Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
    def OTSU_Threshold(self,image):
    # Set total number of bins in the histogram

        bins_num = 256
        
        # Get the image histogram
        
        hist, bin_edges = np.histogram(image, bins=bins_num)
    
        # Get normalized histogram if it is required
        
        #if is_normalized:
        
        hist = np.divide(hist.ravel(), hist.max())
        
        
        
        # Calculate centers of bins
        
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
        
        # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
        
        weight1 = np.cumsum(hist)
        
        weight2 = np.cumsum(hist[::-1])[::-1]
    
        # Get the class means mu0(t)
        
        mean1 = np.cumsum(hist * bin_mids) / weight1
        
        # Get the class means mu1(t)
        
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
        
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        
        # Maximize the inter_class_variance function val
        
        index_of_max_val = np.argmax(inter_class_variance)
        
        threshold = bin_mids[:-1][index_of_max_val]
        
        #print("Otsu's algorithm implementation thresholding result: ", threshold)
        return threshold

    #########################################################################
    def ApplyCLAHE(self,gray):
    #https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
        
        gray_img_eqhist=cv2.equalizeHist(gray)
        hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
        clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
        gray_img_clahe=clahe.apply(gray_img_eqhist)
        return gray_img_clahe

    #########################################################################
    def FindLicenseNumber (self,gray, x_offset, y_offset, x_resize, y_resize, \
                        Resize_xfactor, Resize_yfactor, BilateralOption):
    #########################################################################

        
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
        X_resize=x_resize
        Y_resize=y_resize
        
        
        gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
        
        rotation, spectrum, frquency =self.GetRotationImage(gray)
        rotation=90 - rotation
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
            gray=imutils.rotate(gray,angle=rotation)
        
        TabLicensesFounded=[]
        ContLicensesFounded=[]
        
        
        ##########################################################
        #gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    
        
        #print(gray)
        
        X_resize=x_resize
        Y_resize=y_resize
        #print("gray.shape " + str(gray.shape)) 
        Resize_xfactor=1.5
        Resize_yfactor=1.5
        
        rotation, spectrum, frquency =self.GetRotationImage(gray)
        rotation=90 - rotation
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
            gray=imutils.rotate(gray,angle=rotation)
        
        
        TabLicensesFounded=[]
        ContLicensesFounded=[]
        #https://mattmaulion.medium.com/the-digital-image-an-introduction-to-image-processing-basics-fbdf9fd7f462
        from skimage import img_as_uint
        # for this demo, set threshold to average value
        gray1 = img_as_uint(gray > gray.mean())
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum()) 
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
    
        # https://medium.com/@marizombie/computer-vision-interview-convolutional-neural-network-48e4567e4bed
        kernel = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,-476,24,6], [4,16,24,16,4], [1,4,6,4,1]])
        kernel=kernel*(-1)
        kernel=kernel/256
        im = cv2.filter2D(gray, -1, kernel)
        
        text = pytesseract.image_to_string(im, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum()) 
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        
        
        
        # https://medium.com/@sarcas0705/computer-vision-derivative-over-image-e1020354ddb5
        #sobel
        kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
        gray1 = cv2.filter2D(gray, -1, kernel)
        kernel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
        gray2 = cv2.filter2D(gray, -1, kernel) 
        
        gray1=gray1+gray2
    
        gray1 = cv2.threshold(gray1, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]   
        gray1= cv2.GaussianBlur(gray1, (5,5), 0)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum()) 
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
    
        #https://towardsdatascience.com/morphological-operations-for-image-preprocessing-in-opencv-in-detail-15fccd1e5745
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
        blackhat1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_BLACKHAT, kernel1)
        text = pytesseract.image_to_string(blackhat1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            TabLicensesFounded, ContLicensesFounded =self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        kernel = np.ones((3,3),np.float32)/90
        gray1 = cv2.filter2D(gray,-1,kernel)   
        #gray_clahe = cv2.GaussianBlur(gray, (5, 5), 0) 
        gray_img_clahe=self.ApplyCLAHE(gray1)
        
        th=self.OTSU_Threshold(gray_img_clahe)
        max_val=255
        
        ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
        
        text = pytesseract.image_to_string(o3, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum()) 
        text=self.ProcessText(text)
        if self.ProcessText(text) != "":
            TabLicensesFounded, ContLicensesFounded =self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        #   Otsu's thresholding
        # ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    
        # text = pytesseract.image_to_string(gray1, lang='eng',  \
        # config='--psm 6 --oem 3')
        # text = ''.join(char for char in text if char.isalnum()) 
        # text=self.ProcessText(text)
        # if self.ProcessText(text) != "":
        #     self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        #     print("Text 6 "+text)
        ####################################################
        # experimental formula based on the brightness
        # of the whole image 
        ####################################################
    
        # SumBrightness=np.sum(gray)  
        # threshold=(SumBrightness/177600.00) 
            
        # for z in range(4,8):
        # #for z in range(8,8):
        #     kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
        #     gray1 = cv2.filter2D(gray, -1, kernel)
                    
            
        #     text = pytesseract.image_to_string(gray1, lang='eng',  \
        #         config='--psm 6 --oem 3')
        #     text = ''.join(char for char in text if char.isalnum()) 
        #     text=self.ProcessText(text)
        #     if self.ProcessText(text) != "":
        #         self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        #         print("Text 7 "+text)
        # gray_img_clahe=self.ApplyCLAHE(gray)
        
        
        # ###################################################################
        # # ANTES
        # th=self.OTSU_Threshold(gray_img_clahe)
        # max_val=255
        
        threshold=self.ThresholdStable(gray)
        
        # ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
        # #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
        # text = pytesseract.image_to_string(gray1, lang='eng',  \
        # config='--psm 6 --oem 3')
        # text = ''.join(char for char in text if char.isalnum())
        # #if Detect_Spanish_LicensePlate(text)== 1:
        # if self.ProcessText(text) != "":
        #     self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)         
        # print("Text 8 "+text)
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
        #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        text=self.ProcessText(text)
        #if Detect_Spanish_LicensePlate(text)== 1:
        if self.ProcessText(text) != "":
            self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        
        ####################################################
        # experimental formula based on the brightness
        # of the whole image 
        ####################################################
        
        SumBrightness=np.sum(gray)  
        threshold=(SumBrightness/177600.00) 
        
        #####################################################
        
        # ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
        # #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
        # text = pytesseract.image_to_string(gray1, lang='eng',  \
        # config='--psm 6 --oem 3')
        # text = ''.join(char for char in text if char.isalnum())
        # print("Text 10 "+text)
        # #if Detect_Spanish_LicensePlate(text)== 1:
        # self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        
        # for z in range(5,10):
        #     if z==6:continue
        #     if z==10:continue
        #     kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
        #     gray1 = cv2.filter2D(gray, -1, kernel)
        #     #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)       
        #     text = pytesseract.image_to_string(gray1, lang='eng',  \
        #     config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
            
        #     text = ''.join(char for char in text if char.isalnum())
        #     print("Text 11 "+text)
        #     #if Detect_Spanish_LicensePlate(text)== 1:
        #     self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            
        
    
        
        # for z in range(10,12):
        
        #     kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
        #     gray1 = cv2.filter2D(gray, -1, kernel)
        #     gray1 = cv2.GaussianBlur(gray1, (1, 1), 2)
        #     text = pytesseract.image_to_string(gray1, lang='eng',  \
        #     config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
            
        #     text = ''.join(char for char in text if char.isalnum())
        #     print("Text 12 "+text)
        #     #if Detect_Spanish_LicensePlate(text)== 1:
        #     self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        
        
        #https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438
        
        # bilateral = cv2.bilateralFilter(gray,9,75,75)
        # median = cv2.medianBlur(bilateral,3)
        
        # adaptive_threshold_mean = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #     cv2.THRESH_BINARY,11,2)
            
        # text = pytesseract.image_to_string( adaptive_threshold_mean , lang='eng',  \
        #     config='--psm 6 --oem 3') 
        # text = ''.join(char for char in text if char.isalnum())
        # #if Detect_Spanish_LicensePlate(text)== 1:
        # print("Text 13 "+text)
        # self.ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)     
                
            
        ################################################################
        return TabLicensesFounded, ContLicensesFounded

    ########################################################################

    # COPIED FROM https://programmerclick.com/article/89421544914/
    def gamma_trans (self,img, gamma): # procesamiento de la función gamma
            gamma_table = [np.power (x / 255.0, gamma) * 255.0 for x in range (256)] # Crear una tabla de mapeo
            gamma_table = np.round (np.array (gamma_table)). astype (np.uint8) #El valor del color es un número entero
            return cv2.LUT (img, gamma_table) #Tabla de búsqueda de color de imagen. Además, se puede diseñar un algoritmo adaptativo de acuerdo con el principio de homogeneización de la intensidad de la luz (color).
    def nothing(self,x):
        pass


    def ProcessText(self,text):
        if text is None: return ""
        if len(text)  > 7:
            return text[-7:]
        else:
            return text   

    def ApendTabLicensesFounded (self,TabLicensesFounded, ContLicensesFounded, text):
        SwFounded=0
        for i in range( len(TabLicensesFounded)):
            if text==TabLicensesFounded[i]:
                ContLicensesFounded[i]=ContLicensesFounded[i]+1
                SwFounded=1
                break
        if SwFounded==0:
            TabLicensesFounded.append(text) 
            ContLicensesFounded.append(1)
        return TabLicensesFounded, ContLicensesFounded


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c




          


      
                 
        