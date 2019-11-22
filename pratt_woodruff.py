import math
import numpy as np
import matplotlib.pyplot as plt

### Original plot using factorial instead of gamma
def getBinomialProbUsingFact(n, k, p):
    lnP = math.log(math.factorial(n)) - math.log(math.factorial(k)) - math.log(math.factorial(n-k)) + k*math.log(p) + (n-k)*math.log(1-p)
    # print("lnP: ", lnP)
    prob = math.exp(lnP)
    # print("p: ", p)
    return prob

### n! = gamma(n+1)
### ln(n!) = lngamma(n+1)
### n! = e**lngamma(n+1)
### ln(n!) = ln(e**lngamma(n+1))
### ln(n!) = lngamma(n+1)*ln(e) = lngamma(n+1)
def getBinomialProbUsingGamma(n, k, p):
    lnP = math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1) + k*math.log(p) + (n-k)*math.log(1-p)
    # print("lnP: ", lnP)
    prob = math.exp(lnP)
    # print("p: ", p)
    return prob

### Create function binpmf(n, p)
def binpmf(n, k, p):
    prob = getBinomialProbUsingGamma(n, k, p)
    return prob

def plotProbs():
    n = 60000
    p = 1/5
    minPlot = 11000
    maxPlot = 13000
    supp = list(range(minPlot, maxPlot))
    numGuesses = np.array(supp)
    probCorrect = np.zeros(maxPlot-minPlot)
    for k in range(minPlot, maxPlot):
        prob = binpmf(n, k, p)
        probCorrect[k-minPlot] = prob
    plt.bar(numGuesses, probCorrect)
    plt.xlabel('Prob. Correct')
    plt.ylabel('Prob. Correct')
    plt.title('Plot of Number of Guesses VS Prob. Correct')
    plt.show()

### Using factorial
# getBinomialProbUsingFact(60000, 12489, 1/5)
### Using gamma
# getBinomialProbUsingGamma(60000, 12489, 1/5)
