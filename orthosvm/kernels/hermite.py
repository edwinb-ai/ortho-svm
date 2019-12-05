import cppimport
from time import time


hermite = cppimport.imp("hermite")
for i in range(51):
    start = time()
    print(hermite.hermite(5.0, i))
    stop = time()
    print("Exec time: {}".format(stop - start))
