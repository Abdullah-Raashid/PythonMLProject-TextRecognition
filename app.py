import pygame, sys
from pygame.locals import *
from pygame import image
import numpy as np
from keras.models import load_model
import cv2
from tensorflow.python.keras.backend import constant
from numpy.lib.type_check import imag
from numpy import testing

BOUNDARYINC=5
WINDOWSIZEX=640
WINDOWSIZEY=480

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)


IMAGESAVE = False
PREDICT = True

image_count = 1

MODEL = load_model("/Users/abdullahraashid/Documents/PythonML/DigitRecognition/savemodel.h5")

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine", 10:"Ten"}
#Initialize pygame

pygame.init()

FONT = pygame.font.Font("./MovistarTextRegular.ttf",18)
DISPLAYSURF= pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

pygame.display.set_caption("Digit board")

iswriting = False

number_xcord=[]
number_ycord=[]

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting=True

        # Once done with writing, focus on model

        if event.type == MOUSEBUTTONUP:
            iswriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC,0), min(WINDOWSIZEX,number_xcord[-1]+BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0]-BOUNDARYINC,0), min(number_ycord[-1]+BOUNDARYINC,WINDOWSIZEX)

            number_xcord=[]
            number_ycord=[]

            img_array = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_x:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_count +=1

            #incorporating machine learning
            if PREDICT:
                image = cv2.resize(img_array, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255
                
                # gets the label after the prediction
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface = FONT.render(label,True, RED,WHITE)
                textRectObj = testing.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)
             
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)
        
        pygame.display.update()
            
