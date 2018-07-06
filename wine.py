import classification as cf
import numpy as np
from collections import Counter

data = []
groupNums = []
dataDim = 11 
sampleSize = 4893 
totalGroup = 6

inputDat = open('wine', 'r')
count = 0
for line in inputDat:
    count += 1
    rawDat = line.strip().split('\t')
    groupNums.append(int(rawDat[-1]))
    rawDat = map(float, rawDat[0:dataDim]) #last column denotes category
    data.append(rawDat)
inputDat.close()
data = np.array(data)
groupSizes = np.array(Counter(groupNums).values())

centerIndex = cf.initializeCenter(totalGroup, sampleSize)
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
print count
print cf.totalClassError(newGroup, groupSizes)
