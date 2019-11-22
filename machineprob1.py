#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Machine problem 1
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


### 1.)
### a.)

# Returns the value of N(S=heads)
def getN1000Sn(nHeads):
    # Simulate 1000 repetitions of tossing coins 100 times
    h = np.random.randint(2, size=(1000, 100))
    rowSum = np.sum(h, axis=1)
    totalTrialsEqualN = len(rowSum[rowSum == nHeads])
    meanHeads = rowSum.mean()
    stdHeads = rowSum.std()
    return totalTrialsEqualN, meanHeads, stdHeads

headTrialArr = np.zeros((12, 3)).astype(float)
headMeansArr = np.zeros((12, 3)).astype(float)
headStdArr = np.zeros((12, 3)).astype(float)
aveHeadsPerGroup = np.zeros(12)
numTrials = np.zeros(12)
g1N = 33
### Create three groups
for i in range(0, 12):
    headTrialArr[i, 0], headMeansArr[i, 0], headStdArr[i, 0] = getN1000Sn(g1N) # First group
    headTrialArr[i, 1], headMeansArr[i, 1], headStdArr[i, 1] = getN1000Sn(g1N+1) # Second group
    headTrialArr[i, 2], headMeansArr[i, 2], headStdArr[i, 2] = getN1000Sn(g1N+2) # Third group
    aveHeadsPerGroup[i] = g1N+1
    numTrials[i] = headTrialArr[i].sum()
    print("N1:{0:.2f} + N2:{1:.2f} + N3:{2:.2f} = Sum:{3:.2f}"
          .format(headTrialArr[i, 0],
                  headTrialArr[i, 1],
                  headTrialArr[i, 2],
                  headTrialArr[i].sum()))
    g1N += 3


# In[3]:


### b.) Plot
plt.bar(aveHeadsPerGroup, numTrials)
plt.xlabel('Average Heads')
plt.ylabel('Number of Trials')
plt.title('Heads per Group VS Number of Trials')
plt.show()


# In[4]:


### The graph resembles a bell curve as most of the outcomes of the experiment has an outcome of approx. 50 heads


# In[5]:


### c.)
### Compute probability
### X = can also be interpreted as number of trials with heads Sh over total trials
prob_numTrials = numTrials / numTrials.sum()
expectedVal = headMeansArr.mean()
std = headStdArr.mean()
print("Test1")
for i in range(0, len(aveHeadsPerGroup)):
    print("Probability of getting {0:d} heads: {1:.2f}".format(int(aveHeadsPerGroup[i]), prob_numTrials[i]))
print("ExpectedVal: {0:.5f}".format(expectedVal))
print("STD (averaged): {0:.5f}".format(std))


# In[6]:


### Execute for 19 more tries
for testN in range(0, 19):
    g1N = 33
    for i in range(0, 12):
        headTrialArr[i, 0], headMeansArr[i, 0], headStdArr[i, 0] = getN1000Sn(g1N) # First group
        headTrialArr[i, 1], headMeansArr[i, 1], headStdArr[i, 1] = getN1000Sn(g1N+1) # Second group
        headTrialArr[i, 2], headMeansArr[i, 2], headStdArr[i, 2] = getN1000Sn(g1N+2) # Third group
        numTrials[i] = headTrialArr[i].sum()
        aveHeadsPerGroup[i] = g1N+1
        g1N += 3
    prob_numTrials = numTrials / numTrials.sum()
    expectedVal = headMeansArr.mean()
    std = headStdArr.mean()
    print("Test{0:d}".format(testN+2))
    for i in range(0, len(aveHeadsPerGroup)):
        print("Probability of getting {0:d} heads: {1:.2f}".format(int(aveHeadsPerGroup[i]), prob_numTrials[i]))
    print("ExpectedVal: {0:.5f}".format(expectedVal))
    print("STD (averaged): {0:.5f}".format(std))
    print()


# In[7]:


### 2.)
### Component actual state random sampled from [0,1)
### Example Component Initialization: componentsState = np.random.rand((100))


# In[8]:


###
### The probability that the system is function is (Π - multiplication of reliabilities):
### P(components100) = Π P(comp1to40) + 1 - Π (1 - comp41to50) + Π P(comp51to80) + 1 - Π (1 - comp81to100)

### 2.1)
### Computes reliability of partial components
### componentsState - stores reliability of each component, size=100
### startIdx - start index where the components are in series/parallel
### endIdx - end index where the series/parallel ends + 1 (up to  <= 1-len)
def computeRel(componentsState, startIdx, endIdx, method):
    if method=='S': # Series
        return componentsState[startIdx:endIdx].prod()
    elif method=='P':
        return 1 - (1 - componentsState[startIdx:endIdx]).prod()
    
### Function to compute the whole system reliability
### componentsState - stores reliability of each component, size=100
### Returns the system probability and the reliability probability of (0)1-40, (1)41-50, (2)51-80, (3)81-100
def system_operational(componentsState):
    ### Compute 1-40 comp. reliability (series)
    prob1to40 = computeRel(componentsState, 0, 40, 'S')
    ### Compute 41-50 comp. reliability (parallel)
    prob41to50 = computeRel(componentsState, 40, 50, 'P')
    ### Compute 51-80 comp. reliability (series)
    prob51to80 = computeRel(componentsState, 50, 80, 'S')
    ### Compute 81-100 comp. reliability (parallel)
    prob81to100 = computeRel(componentsState, 80, 100, 'P')
    probTopSeries = prob1to40 * prob41to50
    probBottomSeries = prob51to80 * prob81to100
    totalReliability = 1 - ((1 - probTopSeries) * (1 - probBottomSeries))
    return totalReliability, prob1to40, prob41to50, prob51to80, prob81to100, probTopSeries, probBottomSeries

### (I.) For a series system, the component having the least reliability (or greatest q) should be replaced to improve
### the series system. For a parallel system, the component with the most reliability should be
### replaced to improve parallel system.

### These two functions returns the index of the component that needs to be replaced for series and parallel 
### systems, respectively, as stated in (I.)

### For series...
### wholeComp - whole 100 numpy array components (or can be other series numpy arrays)
### startIdx - start index where the components are in series
### endIdx - end index where the series ends + 1 (up to  <= 1-len)
def getSeriesCompToReplace(wholeComp, startIdx, endIdx):
    seriesComp = wholeComp[startIdx:endIdx]
    sortedIdx = np.argsort(seriesComp) # returns indexes by elements ascending order
    return startIdx + sortedIdx[0]
### For parallel, return idx with largest reliability p
def getParallelCompToReplace(wholeComp, startIdx, endIdx):
    seriesComp = wholeComp[startIdx:endIdx]
    sortedIdx = np.argsort(seriesComp) # returns indexes by elements ascending order
    return startIdx + sortedIdx[len(sortedIdx) - 1]

### 2.2) Replace 20 components with ultra-reliable
### wholeComp - the 100 numpy array components
### totalN - a total of 20 components are needed to be replaced
### ultraRel - the ultra reliable components' failure rate are to be replaced with 1/10 (default) of its original
def replaceCompWithUltra(wholeComp, totalN=20, ultraRel=1/10, randomReplace=False):
    if randomReplace: ### Replace totalN random components
        randomIdxLi = list(range(0, 100))
        random.shuffle(randomIdxLi) #shuffle method
        for i in range(0, totalN):
            ### Get idx to replace
            idxToReplace = randomIdxLi[i]
            wholeComp[idxToReplace] = 1 - ((1 - wholeComp[idxToReplace]) * ultraRel)
            totalReliability, _, _, _, _, _, _ = system_operational(wholeComp)
            print("Component being replaced is (1-100): ", idxToReplace+1)
            print("System Reliability: ", totalReliability)
    else:
        for i in range(0, totalN):
            # Compute reliability of comp (0)1-40, (1)41-50, (2)51-80, (3)81-100 - A SERIES CIRCUIT
            totalReliability, prob1to40, prob41to50, prob51to80, prob81to100, probTopSeries, probBottomSeries = system_operational(wholeComp)
            wholeParallelArr = np.array([probTopSeries, probBottomSeries])
            ### Get system parallel idx part to replace: (0)1-50, (1)51-100
            systemParallelPartToReplace = getParallelCompToReplace(wholeParallelArr, 0, len(wholeParallelArr)) # Return 0-1
            startIdx = None
            endIdx = None
            compIdxToReplace = None
            if systemParallelPartToReplace == 0:
                topSeriesArr = np.random.rand(41)
                for topSeriesCompNum in range(0, 40):
                    topSeriesArr[topSeriesCompNum] = wholeComp[topSeriesCompNum]
                topSeriesArr[len(topSeriesArr) - 1] = prob41to50
                topPartToReplace = getSeriesCompToReplace(topSeriesArr, 0, len(topSeriesArr)) # Return 0-41
                if topPartToReplace >= 0 and topPartToReplace < 40:
                    compIdxToReplace = topPartToReplace
                elif topPartToReplace == 40:
                    startIdx = 40
                    endIdx = 50
                    compIdxToReplace = getParallelCompToReplace(wholeComp, startIdx, endIdx)
            elif systemParallelPartToReplace == 1:
                bottomSeriesArr = np.random.rand(31)
                for bottomSeriesCompNum in range(0, 30):
                    bottomSeriesArr[bottomSeriesCompNum] = wholeComp[bottomSeriesCompNum+50]
                bottomSeriesArr[len(bottomSeriesArr) - 1] = prob81to100
                bottomPartToReplace = getSeriesCompToReplace(bottomSeriesArr, 0, len(bottomSeriesArr)) # Return 0-31
                if bottomPartToReplace >= 0 and bottomPartToReplace < 30:
                    compIdxToReplace = bottomPartToReplace + 50
                elif bottomPartToReplace == 30:
                    startIdx = 80
                    endIdx = 100
                    compIdxToReplace = getParallelCompToReplace(wholeComp, startIdx, endIdx)
            wholeComp[compIdxToReplace] = 1 - ((1 - wholeComp[compIdxToReplace]) * ultraRel)
            # Compute new reliability
            totalReliability, _, _, _, _, _, _ = system_operational(wholeComp)
            print("Component being replaced is (1-100): ", compIdxToReplace+1)
            print("System Reliability: ", totalReliability)


# In[9]:


def replaceWithUltraReliable(randomReplace=False):
    ### Initialize components reliability
    totalUltra = 20
    qa = 0.275
    qb = qa/0.67
    ra = 1-qa ### Components 1-50 reliability
    rb = 1-qb ### Components 51-100 reliability
    ultraRel=0.1 ### 1/10 of q
    wholeComp = np.random.rand(100)
    wholeComp[0:50] = ra
    wholeComp[50:100] = rb
    print("Initial Component Values: ", wholeComp)
    totalReliability, _, _, _, _, _, _ = system_operational(wholeComp)
    print()
    print("System Reliability: ", totalReliability)
    replaceCompWithUltra(wholeComp, totalUltra, ultraRel, randomReplace)


# In[10]:


replaceWithUltraReliable()
### As you can see, System Reliability is increasing with every ultra reliable component replacement
### Even if you run replaceWithUltraReliable() 100 times, the components needed to replaced will still be in the same order


# In[11]:


replaceWithUltraReliable(randomReplace=True)
### Replace random components (totalN=20) and print out reliability of system
### Randomly replacing components with ultra-reliable ones is a bit more costlier


# In[12]:


### 3.) Pratt-Woodruff experiment
from pratt_woodruff import plotProbs
### pratt_woodruff.py for more details
plotProbs()


# In[ ]:





# In[13]:


### 4.) Write zipfunction
def calculateC(n):
    unnormalizedProb = np.random.rand(n)
    xVals = np.random.rand(n)
    for x in range(1, n+1):
        xVals[x-1] = x
        unnormalizedProb[x-1] = 1/x
    ### Normalize to pmf
    normalizedProb = unnormalizedProb / unnormalizedProb.sum()
    ### First calculate c(n)
    c = normalizedProb * xVals
    return c[0], normalizedProb

### Returns max k that provides the k most popular pages having probability <= some {threshold}
def getMaxK(n, c, threshold):
    cumulativeP = 0
    for x in range(0, n):
        cumulativeP += c / (x+1)
        if (cumulativeP > threshold): return (x) ### Use the lower of the two, (cumulativeP <= threshold)
    return -1
### Return k, which is less than or equal to n
### n = 1 to 1000
def zipfunc(n, thresh):
    c, normalizedProb = calculateC(n)
    maxK = getMaxK(n, c, thresh)
    return maxK


# In[14]:


### n = 1000
n = 1000

print("Cache 70% most popular pages")
threshold = 0.7
for i in range(n):
    calculatedMaxK = zipfunc(i+1, threshold)
    print("For n={0:d}, maxK={1:d}".format(i+1, calculatedMaxK))
print()


# In[15]:


print("Cache 80% most popular pages")
threshold = 0.8
for i in range(n):
    calculatedMaxK = zipfunc(i+1, threshold)
    print("For n={0:d}, maxK={1:d}".format(i+1, calculatedMaxK))
print()


# In[16]:


print("Cache 90% most popular pages")
threshold = 0.9
for i in range(n):
    calculatedMaxK = zipfunc(i+1, threshold)
    print("For n={0:d}, maxK={1:d}".format(i+1, calculatedMaxK))

