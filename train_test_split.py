import numpy as np
import cv2
import pickle

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Dur'
    elif classNo == 1:
        return 'Kasisli Yol'
    elif classNo == 2:
        return 'Yaya Gecidi'
    elif classNo == 3:
        return 'Dikkat'
    elif classNo == 4:
        return 'Vahsi Hayvan Cikabilir'
    elif classNo == 5:
        return 'Yol Ver'
    elif classNo == 6:
        return 'Tasit Trafigine Kapali Yol'
    elif classNo == 7:
        return 'sol'
    elif classNo == 8:
        return 'sag'
    elif classNo == 9:
        return 'isikli isaret cihazi'
    elif classNo == 10:
        return 'Donel Kavsak'
    elif classNo == 11:
        return 'İleri-Sol'
    elif classNo == 12:
        return 'İleri-Sag'
    elif classNo == 13:
        return 'Buzlanma'
    elif classNo == 14:
        return 'Azami Hiz Sinirlamasi'
    elif classNo == 15:
        return 'Sola Tehlikeli Viraj'
    elif classNo == 16:
        return 'Saga Tehlikeli Viraj'
    elif classNo == 17:
        return 'Tanimsiz'
    elif classNo == 18:
        return 'Kamyon Giremez'
    elif classNo == 19:
        return 'Ondeki Araci Gecmek Yasaktir'
    elif classNo == 20:
        return 'Giris Olmayan Yol'
    elif classNo == 21:
        return 'Tali Yol Kavsagi'
    elif classNo == 22:
        return 'Askeri Hiz Sonu'
    elif classNo == 23:
        return 'Ana Yol'
    elif classNo == 24:
        return 'Azami Hiz Sinirlamasi Sonu'
    elif classNo == 25:
        return 'Bisiklet'
    elif classNo == 26:
        return 'Kontrollu'
    elif classNo == 27:
        return 'kontrolsuz'
while True:

    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("İslenmis Goruntu", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "SINIF: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "OLASILIK: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        print(getCalssName(classIndex))
    cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break