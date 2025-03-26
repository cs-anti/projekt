import cv2
import numpy as np
import os
import utlis

########################################################################
webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg = 480
########################################################################

utlis.initializeTrackbars()
cv2.waitKey(100)  # Várunk egy kicsit, hogy a trackbar ablak megjelenjen

count = 0

# Ellenőrizzük, hogy létezik-e a mentési mappa
if not os.path.exists("Scanned"):
    os.makedirs("Scanned")

while True:

    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Nem sikerült képet olvasni a kamerából.")
            break
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Nem található kép: {pathImage}")
            break

    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = utlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20,
                                        20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warped", "Warp Gray", "Adaptive Thresh"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and biggest.size != 0:
        cv2.imwrite(f"Scanned/myImage_{count}.jpg", imgWarpColored)
        cv2.rectangle(stackedImage, (150, 150), (460, 200), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (160, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
