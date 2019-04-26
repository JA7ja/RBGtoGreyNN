import numpy as np
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
    vModel.add(keras.layers.Dense(1, activation='relu'));
    vModel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #Model training
    vModel.fit(vTrainingData,vTrainingGreyValues,epochs=20)
    #Evaluate the testing set from trained model
    vResults=vModel.evaluate(vTestingData,vTestingGreyValues);
    print("Testing Accuracy is: %f" % vResults[1]);

def main():
    pixelValueConversion()

if __name__ == "__main__":
    main();
