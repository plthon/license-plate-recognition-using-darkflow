import datetime
import re
import time
import cv2
import numpy as np
import pytesseract
import sys
from PIL import Image

from darkflow.net.build import TFNet

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

options = {"pbLoad": "../src/inference_graph.pb", "metaLoad": "yolo-plate.meta", "gpu": 0.9}
yoloPlate = TFNet(options)


def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
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

    if counter % 60 == 0:
        licensePlate = []
        try:
            predictions = yoloPlate.return_predict(frame)
            firstCropImg = firstCrop(frame, predictions)
            secondCropImg = secondCrop(firstCropImg)

            secondCropImg = 255 - (cv2.cvtColor(secondCropImg, cv2.COLOR_BGR2GRAY))

            cv2.imshow('License Plate', secondCropImg)

            tessImg = Image.fromarray(secondCropImg)

            text = pytesseract.image_to_string(tessImg, lang='eng',
                                               config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

            text = outputProcess(text)

            if (text != ''):
                ts = time.time()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Tesseract:", text)

        except Exception as e:
            pass

    counter += 1
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()