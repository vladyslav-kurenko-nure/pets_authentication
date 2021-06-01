import cv2

def CreateRGBHist(image):
    image = cv2.resize(image, (256, 256))
    histrArr = []
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        histrArr.append(histr)

    return histrArr
