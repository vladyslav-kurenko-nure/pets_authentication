import cv2

def CreateHOGDes(image):
    image = cv2.resize(image, (256, 256))
    hog = cv2.HOGDescriptor()
    winstride = (8, 8)
    padding = (8, 8)
    locations = ((10,20),)
    d = hog.compute(image, winstride, padding, locations)

    return d
