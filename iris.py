import classification as cf
import numpy as np
import time

startTime = time.time()
data = []
dataDim = 4
sampleSize = 50
totalGroup = 3

inputDat = open('irisData', 'r')
for line in inputDat:
    rawDat = line.strip().split(',')
    rawDat = map(float, rawDat[slice(1, 3)]) #last column denotes category
    data.append(rawDat)
inputDat.close()
data = np.array(data)
index = list(range(totalGroup*sampleSize))
group = np.concatenate((np.zeros(sampleSize), np.ones(sampleSize), 2*np.ones(sampleSize)), axis
        = 0) 

centerIndex = cf.initializeCenter(totalGroup, totalGroup*sampleSize)
centers = data[centerIndex, :] 
newGroup = cf.classify(data, centers)
count = 0
while True:
    count += 1
    oldGroup = newGroup[:]
    centers = cf.updateCenter(data, newGroup, totalGroup)
    newGroup = cf.classify(data, centers)
    if np.array_equal(oldGroup, newGroup):
        break
groupSizes = sampleSize * np.ones(totalGroup)
print cf.totalClassError(newGroup, groupSizes)
print count
totalTime = time.time() - startTime
print totalTime
                                           
