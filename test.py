import classification as cf
import numpy as np

dataDim = 4
sampleSize = 50
totalGroup = 3

totalSample = totalGroup * sampleSize

a = np.random.randint(1,20, size=(10,3))
print a
print "\n"

centers = a[[2,7,9], :]
print centers
print "\n"

b = cf.classify(a, centers)
print b
print np.size(b)
print "\n"

c = cf.updateCenter(a, b, totalGroup)

print c
print "\n"

