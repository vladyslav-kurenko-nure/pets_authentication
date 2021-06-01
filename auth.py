import detector, camera, classifier
import cv2

c = camera.Connect(r'.\img\test\verify_video_tyhon.mp4')
#c = camera.Connect(0)

cf = 1
fps = round(c.get(cv2.CAP_PROP_FPS) % 100)

while c.isOpened():
    fl, cimg = camera.Capture(0, c)
    if not fl:
        break
    cv2.imshow("", cv2.resize(cimg, (512, 512)))
    if cv2.waitKey(10) == ord('q'):
        break

    if cf == fps:
        f_image1, out_image1, k1 = detector.SearchCat(r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalcatface.xml', cimg)
        f_image2, out_image2, k2 = detector.SearchCat(r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalcatface_extended.xml', cimg)
        f_image3, out_image3, k3 = detector.SearchHuman(r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml', cimg)
        f_image4, out_image4, k4 = detector.SearchHuman(r'.\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml', cimg)

        if detector.Check(k1, k2, k3, k4) == False:
           continue

        for f_image in f_image1:
            for cl in [r'.\src\ml\model_Timur.xml', r'.\src\ml\model_Tyhon.xml']:
                if classifier.Classify(cl, f_image) == 1:
                    print("Welcome", cl)
                    cv2.putText(out_image1, "Welcome", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    break

        cf = 1

        cv2.imshow("out_image1", out_image1)

    else:
        cf += 1

c.release()
cv2.destroyAllWindows()
