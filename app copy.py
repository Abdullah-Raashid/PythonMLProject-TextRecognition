import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

BOUNDARYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
PREDICT = True

image_count = 1

MODEL = load_model("/Users/abdullahraashid/Documents/PythonML/DigitRecognition/savemodel.h5")

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

pygame.init()

FONT = pygame.font.Font("./MovistarTextRegular.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit board")

iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            img_array = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{image_count}.png", img_array)
                image_count += 1

            if PREDICT:
                image = cv2.resize(img_array, (28, 28))
                image = image / 255.0
                image = image.reshape(1, 28, 28, 1)

                label = str(LABELS[np.argmax(MODEL.predict(image))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
