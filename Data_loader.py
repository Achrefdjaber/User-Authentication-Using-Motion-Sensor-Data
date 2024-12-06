'''
Python utilities for HMOG dataset
Five functions are implemented here:
 1) overSampling(): For touch gesture samples, because the length of touch gesture logs is small compared to sensor logs.
 
    - overSampling() function is used for replicating touch gesture samples using the concept of *oversampling*. 
    - Motion and touch data don't have the same length across different sessions and users.
 2) ETLHelper(): ETL stands for: Extract Transform and Load.
 3) ETL().
 4) dataGenerator()
 5) slice(): For slicing the data in an overlapping fashion. 
'''
# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.preprocessing import StandardScaler


'''
overSampling():

A) Parameters:
- touchData: numpy array of touchscreen data extracted from csv files.
- maxLength: integer which denotes the maximum length wanted for the touchData.
- touchData must have the same length as sensor data. 

B) Output:
-  Returns a new touchData array with the same length as sensorData array

C) overSampling uses two methods from random module:
- 1) random.choices(): to generate random list from a given sequence of indices
random.choices() is used if the number of indices >= length of touchData

ex: 
- missignList = random.choices([0, 1, 2], k = 20), here the given list has 3 elements and indices=20
- Length of the list=3 can be less than the number of indices=20. 

- 2) random.sample(): to generate random list from a given sequence of indices
random.sample() is used only if the number of indices < length of touchData

ex:
- missingList = random.sample(range(0,10), 10), given sequence has 10 elements, inidces can be <=10. 
'''
from tensorflow.keras.models import Sequential, load_model


def overSampling(touchData, maxLength):

    missingValues = maxLength - len(touchData)
    if missingValues >= len(touchData):

        touchList = list(range(len(touchData)))
        # missingIndices : generated random indices in range: touchList with max values = missingValues
        missingIndices = random.choices(touchList, k=missingValues) 
        missingData = []
        for i in range(len(missingIndices)):
            missingData.append(touchData[missingIndices[i],:])

        missingData = np.array(missingData)
        touchArray = np.vstack([touchData, missingData])
    else:
        # if the length of the missing values is < to the length of touchData 
        # we use random.sample()
        
        missingIndices = random.sample(range(0,missingValues), missingValues)
        missingData = touchData[missingIndices,:]
        touchArray = np.vstack([touchData, missingData])
        
    return touchArray

"""
ETLHelper takes an input, ex: userPath=0 (userId between 0 and 99), and a session ex: sessionPath=0. 
ETL Helper reads out the sensor csv files with numpy.genfromtxt() function and stack horizontally the files and returns the result
"""


def ETLHelper(userID, sessionPath):
    # define the current working directory
    home = os.getcwd()

    userPath =  home + '/' + str(userID) +'/'
    sessionPath = str(sessionPath) + '/'
    path = Path(userPath, sessionPath)

    os.chdir(path)

    # Load sensor data (acc, gyr, mag)
    acc = np.genfromtxt('Accelerometer.csv', delimiter=',')[:10000,3:6]
    gyr = np.genfromtxt('Gyroscope.csv', delimiter=',')[:10000,3:6]
    mag = np.genfromtxt('Magnetometer.csv', delimiter=',')[:10000,3:6]
    # Compute the norm for each sensor
    acc_norm = np.sqrt((acc[:,0])**2 + (acc[:,1])**2 + (acc[:,2])**2)
    gyr_norm = np.sqrt((gyr[:,0])**2 + (gyr[:,1])**2 + (gyr[:,2])**2)
    mag_norm = np.sqrt((mag[:,0])**2 + (mag[:,1])**2 + (mag[:,2])**2)
    
    # Add a dimension for each sensor: (nrow_, 1 column)
    acc_norm = np.expand_dims(acc_norm, axis=1)
    gyr_norm = np.expand_dims(gyr_norm, axis=1)
    mag_norm = np.expand_dims(mag_norm, axis=1)
    sensorStack = np.hstack([acc_norm, gyr_norm, mag_norm])

    # load touch gesture data
    touchActivity = np.genfromtxt('TouchEvent.csv', delimiter=',')[:10000,[6,7,8,9]]
    if len(touchActivity) < 10000:
        touchActivity = overSampling(touchData=touchActivity, maxLength=10000)

    os.chdir(home)
    
    return sensorStack, touchActivity


"""
ETL() takes a user target ID with and a session list : [].
Using the ETLHelper(), the ETL() function loops through the sessions and returns (or yield) the data by stacking each session vertically.
"""

def ETL(user, session):
    sessionSetSensor = []
    sessionSetTouch = []

    for i in tqdm(range(len(session))):
        s, t =  ETLHelper(userID=user, sessionPath=session[i])
        
        sessionSetSensor.append(s) # append all sensor data
        sessionSetTouch.append(t)  # append all touch data


    sessionSetSensor = np.vstack(sessionSetSensor)
    sessionSetTouch = np.vstack(sessionSetTouch)

    # userLabel = user
    userLabel = np.zeros(sessionSetSensor.shape[0]) + user    #Labeling all the data is required
    yield sessionSetSensor, sessionSetTouch, userLabel


''' 
dataGenerator()
Generate:
- sensorData
- TouchData
- Labels
Append them in a list and stack them

A) Parameters:
- 1) users: Integer value between [0, 99].
- 2) session: Integer value between [0,23].

B) Output:
- 1) sensor: numpy array of sensor logs
- 2) touch: numpy array of touch logs
- 3) labels: ground truth labels for each user. 

'''

def dataGenerator(numUsers, session):
    sensorData = []
    touchData = []
    y = []
    u = 0
    while u <= numUsers:
        print("User ID: ", u)
        for sensor, touch, labels in ETL(user=u, session=range(session)):
            
            print("shape of input sensor", sensor.shape)
            print("shape of touch data", touch.shape)
            print("Shape of labels", labels.shape)
            sensorData.append(sensor)
            touchData.append(touch)
            y.append(labels)
            u +=1

    sensorArray = np.vstack(sensorData)
    touchArray = np.vstack(touchData)
    labelArray = np.hstack(y) 
    # hstack because y is returned in a columnwise fashion
    labelArray = np.expand_dims(labelArray, axis=1)
    sensor_scaler = StandardScaler()
    touch_scaler = StandardScaler()
    sensorScaler = sensor_scaler.fit(sensorArray)
    touchScaler = touch_scaler.fit(touchArray)

    sensorNormalized = sensorScaler.transform(sensorArray)
    touchNormalized = touchScaler.transform(touchArray)

    return sensorNormalized, touchNormalized, labelArray


'''
Implement the slice() fucntion:
A ) Parameters:
- data: input array --> input type: numpy array

- window_length: length of the slicing window --> input type integer

- overlapping: the percentage of overlapping --> input type float [0,1]
'''

def slice(data, window_length, overlapping):
    length_data = len(data)

    number_slices = int(np.floor((length_data - window_length)/(window_length*(1 - overlapping)))+1)

    slices = []
    for i in range(number_slices):

        start = int(i * window_length * (1-overlapping))

        end = start + window_length

        slices.append(data[start:end,:])

    slice_samples = np.array(slices)
    
    
    return slice_samples


def split_datasety(dataset, chunk_size):
    num_chunks = dataset.shape[0] // chunk_size
    chunks = []
    for i in range(0,num_chunks):
            start = i*chunk_size
            end = start + chunk_size
            chunk = dataset[end-1:end, :]
            chunks.append(chunk)
    return np.array(chunks)
