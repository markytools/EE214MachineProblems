#!/usr/bin/env python
# coding: utf-8
### Machine Problem Set 2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### 1.) Temperature
### Calculate covariance between temp. at idx i and j
### t = 32-dim vector of temperatures
### i = index of first temperature
### j = index of second temperature
### if i==j, then this function returns the variance
def covTemps(t, i, j):
    return 36 / (1 + abs(i - j))

### a.)
### t = is a 32-dim vector of temperature values in Fahrenheit
### Calculate prob. that P[Y >= T] = 1 - P[Y <= T]
### In addition to calculating the probability, this function also returns the daily average temp. and its corresponding min. temp. for all 10000 samples. This is for answering letter b.
def julytemps(t):
    total_samples = 10000
    n = 31
    mu = 80 # E[Ti]
    var = 36 # From covariance formula, for all i=j
    mu = np.full(n, mu)
    var = np.full(n, var)
    temp13_10000 = np.zeros((total_samples, n)) # Initialize
    for i in range(0, total_samples):
        temp13_10000[i] = np.random.normal(mu, var, n)
    min_temperatures = temp13_10000.min(axis=1)
    daily_average_temp_samples = np.mean(temp13_10000, axis=1)
    daily_average_temp = daily_average_temp_samples
    daily_average_temp[daily_average_temp < t] = 0
    daily_average_temp[daily_average_temp >= t] = 1
    prob_daily_ave = daily_average_temp.sum() / daily_average_temp.size
    return prob_daily_ave, daily_average_temp_samples, min_temperatures

### Example prediction, t=80
prob_daily_ave, daily_average_temp_samples, min_temperatures = julytemps(80)
print("Probability of P[Y >= 80]: ", prob_daily_ave)

### b.)
daily_average_temp_samples[daily_average_temp_samples <= 82] = 1
daily_average_temp_samples[daily_average_temp_samples > 82] = 0
min_temperatures[min_temperatures < 72] = 0
min_temperatures[min_temperatures >= 72] = 1
eventA_hit = 0
for idx in range(0, len(min_temperatures)):
    if daily_average_temp_samples[idx] == 1 and min_temperatures[idx] == 1:
        eventA_hit += 1
prob_event_A = eventA_hit / len(min_temperatures)
print("Probability of A = {Y <= 82, min Ti >= 72}: ", prob_event_A) ### Very little to no chance of this event happening


### 2.) Markov Chain
### a.) Construct Markov state diagram
image = mpimg.imread("MarkovChainMP2.png")
imgplot = plt.imshow(image)
plt.show()

### b.)
### Create state probabilities
p00 = 0.95
p01 = 0.04
p02 = 0.01
p10 = 0.05
p11 = 0.9
p12 = 0.05
p20 = 0.05
p21 = 0.05
p22 = 0.9
p_matrix = np.array([[p00, p01, p02],[p10, p11, p12],[p20, p21, p22]])

### Calculate n-step transition
def markovdisk(n):
    return np.linalg.matrix_power(p_matrix, n)

### P(10)
markovdisk(10)

### P(100)
markovdisk(100)

### P(1000)
markovdisk(1000)


### 3.) See PDF file...


### 4.) a.) See PDF File

### b.)
from numpy import sin
from matplotlib import pyplot
# consistent interval for x-axis
x = [x*0.1 for x in range(100)]
# function of x for y-axis
y = sin(x)
# create line plot
plt.plot(x, y)
# show line plot
plt.show()

from random import seed
from random import randint
from matplotlib import pyplot
# seed the random number generator
seed(1)
# names for categories
x = ['red', 'green', 'blue']
# quantities for each category
y = [randint(0, 100), randint(0, 100), randint(0, 100)]
# create bar chart
pyplot.bar(x, y)
# show line plot
pyplot.show()

# example of a histogram plot
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
# seed the random number generator
seed(1)
# random numbers drawn from a Gaussian distribution
x = randn(1000)
# create histogram plot
pyplot.hist(x)
# show line plot
pyplot.show()

# example of a box and whisker plot
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
# seed the random number generator
seed(1)
# random numbers drawn from a Gaussian distribution
x = [randn(1000), 5 * randn(1000), 10 * randn(1000)]
# create box and whisker plot
pyplot.boxplot(x)
# show line plot
pyplot.show()

# example of a scatter plot
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
# seed the random number generator
seed(1)
# first variable
x = 20 * randn(1000) + 100
# second variable
y = x + (10 * randn(1000) + 50)
# create scatter plot
pyplot.scatter(x, y)
# show line plot
pyplot.show()

### c.) See PDF File
### d.) See PDF File

### e.) Sample tolerance value
# parametric tolerance interval
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import sqrt
from scipy.stats import chi2
from scipy.stats import norm
# seed the random number generator
seed(1)
# generate dataset
data = 5 * randn(100) + 50
# specify degrees of freedom
n = len(data)
dof = n - 1
# specify data coverage
prop = 0.95
prop_inv = (1.0 - prop) / 2.0
gauss_critical = norm.ppf(prop_inv)
print('Gaussian critical value: %.3f (coverage=%d%%)' % (gauss_critical, prop*100))
# specify confidence
prob = 0.99
prop_inv = 1.0 - prob
chi_critical = chi2.ppf(prop_inv, dof)
print('Chi-Squared critical value: %.3f (prob=%d%%, dof=%d)' % (chi_critical, prob*100,
dof))
# tolerance
interval = sqrt((dof * (1 + (1/n)) * gauss_critical**2) / chi_critical)
print('Tolerance Interval: %.3f' % interval)
# summarize
data_mean = mean(data)
lower, upper = data_mean-interval, data_mean+interval
print('%.2f to %.2f covers %d%% of data with a confidence of %d%%' % (lower, upper,
prop*100, prob*100))

# plot tolerance interval vs sample size
from numpy.random import seed
from numpy.random import randn
from numpy import sqrt
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib import pyplot
# seed the random number generator
seed(1)
# sample sizes
sizes = range(5,15)
for n in sizes:
    # generate dataset
    data = 5 * randn(n) + 50
    # calculate degrees of freedom
    dof = n - 1
    # specify data coverage
    prop = 0.95
    prop_inv = (1.0 - prop) / 2.0
    gauss_critical = norm.ppf(prop_inv)
    # specify confidence
    prob = 0.99
    prop_inv = 1.0 - prob
    chi_critical = chi2.ppf(prop_inv, dof)
    # tolerance
    tol = sqrt((dof * (1 + (1/n)) * gauss_critical**2) / chi_critical)
    # plot
    pyplot.errorbar(n, 50, yerr=tol, color='blue', fmt='o')
# plot results
pyplot.show()
### Experiments show that the tolerance decreases as the number of samples increases
