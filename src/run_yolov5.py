import datetime
import re
import time
import cv2
import numpy as np
import pytesseract
import sys
from PIL import Image
import torch
# import pyodbc
#
# server = 'tcp:http://server101218176.database.windows.net,1433'
# database = 'CarplateSystem'
# username = 'innovate'
# password = 'bigbrain!21'
#
# cnxn = pyodbc.connect("DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;" +
#                       "SERVER=" + server + ";" +
#                       "DATABASE=" + database + ";" + "UID=" + username + ";" + "PWD=" + password)
#
# cursor = cnxn.cursor()

modelYOLOv5 = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/plthon/PycharmProjects/license-plate-recognition-using-darkflow/src/best.pt')  # default

# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/4.1.1/bin/tesseract'


def firstCrop(img, predictions):
    xtop = int(predictions.iloc[0]['xmin'])
    ytop = int(predictions.iloc[0]['ymax'])
    xbottom = int(predictions.iloc[0]['xmax'])
    ybottom = int(predictions.iloc[0]['ymin'])
    # print(xtop, ytop, xbottom, ybottom)
    firstCrop = img[ybottom:ytop, xtop:xbottom]
    cv2.rectangle(img, (xtop, ytop), (xbottom, ybottom), (0, 255, 0), 3)
    return firstCrop


def secondCrop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if (len(areas) != 0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        secondCrop = img[y:y + h, x:x + w]
    else:
        secondCrop = img
    return secondCrop


def outputProcess(textX):
    text1 = textX.replace('\n', '')
    text2 = text1.strip().upper()
    text3 = re.sub('[\W_]+', '', text2)
    return text3


cap = cv2.VideoCapture('2019_00031.mp4')
counter = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break

    h, w, l = frame.shape

    if counter % 6 == 0:
        licensePlate = []
        try:
            results = modelYOLOv5(frame)
            coordinates = results.pandas().xyxy[0]
            bestConf = coordinates.loc[[coordinates['confidence'].argmax()]]
            firstCropImg = firstCrop(frame, bestConf)
            secondCropImg = secondCrop(firstCropImg)

            secondCropImg = 255 - (cv2.cvtColor(secondCropImg, cv2.COLOR_BGR2GRAY))

            # cv2.imshow('License Plate', secondCropImg)

            tessImg = Image.fromarray(secondCropImg)

            text = pytesseract.image_to_string(tessImg, lang='eng',
                                               config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

            text = outputProcess(text)

            if (text != ''):
                ts = time.time()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Tesseract:", text)
                # cursor.execute("INSERT INTO MainRecord(CarplateNo, Timestamp, longitude, latitude, CarStatus) VALUES(?, ?, ?, ?, ?)",
                #                (text, timestamp, 11.511212, 12.361221, 'NOR'))
                # cnxn.commit()

        except Exception as e:
            # print(e)
            pass

    counter += 1
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cursor.close()
cap.release()
cv2.destroyAllWindows()
sys.exit()
