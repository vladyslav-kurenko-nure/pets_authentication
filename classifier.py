import cv2
import numpy as np
import os.path
import descriptor, rgbhist

def CreateModel(crop_p_path, crop_n_path, new_model_name):
    dhog_p = []
    dhog_n = []
    rgbh_p = []
    rgbh_n = []

    for path_image1 in os.listdir(crop_p_path):
        image = cv2.imread(crop_p_path + r"\\" + path_image1)
        image = cv2.resize(image, (256, 256))
        dhog_p.append(np.array(descriptor.CreateHOGDes(image).flatten()))
        rgbh_p.append(np.concatenate((np.array(rgbhist.CreateRGBHist(image)[0]), np.array(rgbhist.CreateRGBHist(image)[1]), np.array(rgbhist.CreateRGBHist(image)[2]))).flatten())
    for path_image2 in os.listdir(crop_n_path):
        image = cv2.imread(crop_n_path + r"\\" + path_image2)
        image = cv2.resize(image, (256, 256))
        dhog_n.append(np.array(descriptor.CreateHOGDes(image).flatten()))
        rgbh_n.append(np.concatenate((np.array(rgbhist.CreateRGBHist(image)[0]), np.array(rgbhist.CreateRGBHist(image)[1]), np.array(rgbhist.CreateRGBHist(image)[2]))).flatten())

    for dhog_p_el in dhog_p:
        dhog_p_el /= dhog_p_el.max()
    for dhog_n_el in dhog_n:
        dhog_n_el /= dhog_n_el.max()
    for rgbh_p_el in rgbh_p:
        rgbh_p_el /= rgbh_p_el.max()
    for rgbh_n_el in rgbh_n:
        rgbh_n_el /= rgbh_n_el.max()

    labels = np.concatenate((np.ones(len(os.listdir(crop_p_path)), dtype=np.int32), np.zeros(len(os.listdir(crop_n_path)), dtype=np.int32)))
    trainData_dhog = np.concatenate((dhog_p, dhog_n), dtype=np.float32)
    trainData_rgbh = np.concatenate((rgbh_p, rgbh_n), dtype=np.float32)
    trainData = np.matrix(np.concatenate((trainData_dhog, trainData_rgbh), dtype=np.float32, axis=1), dtype=np.float32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
    svm.save(r'.\src\ml\\' + new_model_name)

def Classify(model_path, image):
    svm = cv2.ml.SVM_load(model_path)
    dataHOG = np.array(descriptor.CreateHOGDes(image).flatten())
    dataRGB = np.concatenate((np.array(rgbhist.CreateRGBHist(image)[0]), np.array(rgbhist.CreateRGBHist(image)[1]), np.array(rgbhist.CreateRGBHist(image)[2]))).flatten()
    data = np.matrix(np.concatenate((dataHOG, dataRGB), dtype=np.float32), dtype=np.float32)
    response = svm.predict(data)[1]

    return response

CreateModel(r'.\src\ml\Timur\p', r'.\src\ml\Timur\n', 'model_Timur.xml')
CreateModel(r'.\src\ml\Tyhon\p', r'.\src\ml\Tyhon\n', 'model_Tyhon.xml')
