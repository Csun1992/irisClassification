import numpy as np
from random import sample 
from numpy.linalg import inv
import math
from scipy import stats 

def initializeCenter(totalGroup, totalSample):
    centers = list(sample(range(totalSample), totalGroup)) 
    return centers

def classify(data, centers):
    centerNum = np.size(centers, axis=0)
    dist = aggregateDist(data, centers) 
    classifiedGroup = np.argmin(dist, axis=1) 
    return classifiedGroup
    
def aggregateDist(data, centers):
    dataSize = np.size(data, axis=0)
    centerNum = np.size(centers, axis=0)
    dist = np.zeros((dataSize, centerNum))
    for i in range(centerNum):
        dist[:, i] = distance(data, centers[i, :])
    return dist
    
def distance(data, centers):
    dataSize = np.size(data, axis=0)
    centeredData = centralize(data, centers)
    dist = np.zeros(dataSize)
    for i in range(dataSize):
        dist[i] = centeredData[i, :].dot(centeredData[i,:].T)
    return dist

def centralize(data, center):
    centeredData = data - center
    return centeredData

def updateCenter(data, classifiedGroup, groupNum):
    groupSize, groupSum = splitGroup(data, classifiedGroup, groupNum) 
    dataDim = np.size(data, axis=1)
    newCenter = np.zeros((groupNum, dataDim)) 
    for i in range(groupNum):
        newCenter[i, :]  = map(float, groupSum[i, :])/groupSize[i]
    return newCenter
    
def splitGroup(data, classifiedGroup, groupNum):
    groupSize = np.zeros(groupNum) 
    dataDim = np.size(data, axis=1)
    groupSum = np.zeros((groupNum, dataDim))
    dataSize = np.size(data, axis=0)
    for i in range(dataSize):
        groupSize[classifiedGroup[i]] += 1 
        groupSum[classifiedGroup[i], :] = groupSum[classifiedGroup[i], :] + data[i, :] 
    return groupSize, groupSum

def totalClassError(result, groupSizes):
    groupNum = np.size(groupSizes)
    aggregateGroupSize = [0]
    for i in range(1,groupNum+1):
        aggregateGroupSize.append(reduce(lambda x,y:x+y, groupSizes[0:i]))
    aggregateGroupSize = map(int, aggregateGroupSize)
    groupErrors = np.zeros(groupNum)
    for i in range(groupNum):
        groupErrors[i] = classError(result[aggregateGroupSize[i]:aggregateGroupSize[i+1]])
    return sum(groupErrors)/float(np.size(result))

        
def classError(result):
    size = np.size(result)
    correctVal = stats.mode(result)[0][0]
    error = sum(result!=correctVal)/float(size)
    return error

        
