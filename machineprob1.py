### Machine problem 1
import numpy as np

### 1.)
### a.)

# Returns the value of N(S=heads)
def getN1000Sn(nHeads):
    # Simulate 1000 repetitions of tossing coins 100 times
    h = np.random.randint(2, size=(1000, 100))
    rowSum = np.sum(h, axis=1)
    return len(rowSum[rowSum == nHeads])
