import numpy as np
import cv2
import pandas
import keras.models
import keras.layers
import sklearn.preprocessing
import sklearn.model_selection

def pixelValueConversion():
    #Loading in data using pandas
    vData = pandas.read_csv('pixelVals3.csv', header=None)
    vRawDataSet = vData.values;
    #Splitting up data into features(b, g, r) and grey values
    vNumberOfFeatures = 3;
    vFeatures = vRawDataSet[:, 0:vNumberOfFeatures];
    vGreyValues = vRawDataSet[:, vNumberOfFeatures];
    vTrainingData, vTestingData, vTrainingGreyValues, vTestingGreyValues =\
                   sklearn.model_selection.train_test_split(vFeatures, vGreyValues, test_size = .2)
    vTrainingData = np.array(vTrainingData)
    vTestingData = np.array(vTestingData)
    vTrainingGreyValues = np.array(vTrainingGreyValues)
    vTestingGreyValues = np.array(vTestingGreyValues)
    #Creating the network model using keras
    vModel = keras.models.Sequential();
    vModel.add(keras.layers.Dense(2, input_dim=vTrainingData.shape[1], activation='relu'));
    vModel.add(keras.layers.Dense(3, activation='relu'));
    vModel.add(keras.layers.Dense(1, activation='relu'));
    vModel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #Model training
    vModel.fit(vTrainingData,vTrainingGreyValues,epochs=50)
    #Evaluate the testing set from trained model
    vResults=vModel.evaluate(vTestingData,vTestingGreyValues);
    print("Testing Accuracy is: %f" % vResults[1]);
    return vModel

def convertWithModel(image, model):
    img = image
    i_range = len(img)
    j_range = len(img[0])
    grey = [ [ 0 for j in range(j_range)] for i in range(i_range)]

    for i in range(i_range):
        for j in range(j_range):
            data = np.array([[img[i][j][0], img[i][j][1], img[i][j][2]]])
            val = model.predict(data)[0]
            if val > 255:
                grey[i][j] = 255
            elif val < 0:
                grey[i][j] = 0
            else:
                grey[i][j] = val

    grey = np.array(grey, dtype=np.uint8)
    return grey

def showImg(img):
    cv2.imshow('Display', img)
    cv2.waitKey()

def main():
    model = pixelValueConversion()
    image = cv2.imread('flowers.jpg', 1)
    grey = convertWithModel(image, model)
    showImg(image)
    showImg(grey)


if __name__ == "__main__":
    main();
