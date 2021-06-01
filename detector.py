import cv2

def SearchCat(path_cascade, image):
    image = cv2.resize(image, (512, 512))
    cascade = cv2.CascadeClassifier(path_cascade)
    b_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #hs_image = cv2.equalizeHist(b_image)
    f_obj = cascade.detectMultiScale(
        b_image,
        scaleFactor=1.01,
        minNeighbors=2,
        minSize=(75, 75)
    )

    f_image = []
    for (x, y, w, h) in f_obj:
        f_image.append(image[y: y + h, x: x + w])

    out_image = image.copy()
    for (x, y, w, h) in f_obj:
        cv2.rectangle(out_image, (x, y), (x + w, y + h), (0, 250, 0), 2)

    k = len(f_obj)

    return f_image, out_image, k

def SearchHuman(path_cascade, image):
    cascade = cv2.CascadeClassifier(path_cascade)
    b_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_obj = cascade.detectMultiScale(
        b_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    f_image = []
    for (x, y, w, h) in f_obj:
        f_image.append(image[y: y + h, x: x + w])

    out_image = image.copy()
    for (x, y, w, h) in f_obj:
        cv2.rectangle(out_image, (x, y), (x + w, y + h), (0, 250, 0), 2)

    k = len(f_obj)

    return f_image, out_image, k

def Check(k1, k2, k3, k4):
    if k1 > 0:
        hdc1 = True
    else:
        hdc1 = False
    if k2 > 0:
        hdc2 = True
    else:
        hdc2 = False
    if k3 > 0:
        hdf1 = True
    else:
        hdf1 = False
    if k4 > 0:
        hdf2 = True
    else:
        hdf2 = False

    hdc = hdc1 or hdc2
    hdf = hdf1 or hdf2

    return hdc and (not hdf)
