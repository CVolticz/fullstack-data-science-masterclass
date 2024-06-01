# Databricks notebook source
# MAGIC %md
# MAGIC # Basic Statistics
# MAGIC Before we can dive into the fun of data science, we need to lay some statistical groundwork. Some of these could be a review if you've ever taken a statistics course previously. Regardless, it'll be worth to create a good foundation that we can continously build upon. 
# MAGIC
# MAGIC In statistics, there are two school of thoughts, Frequentist and Bayesian. Due to the nature of Bayesian statistics, where historically only being able to address a few cases when a priors (a probability the represents what is originally belived before new evidence is introduced) were known, it has been neglected over the years. For that reason, we will start with Frequentist statisitcs and build into Bayesian theory.
# MAGIC
# MAGIC ## Basic Notation
# MAGIC Let's start our discussion by aligning ourselves with some basic notation in a way that make it easy for us to talk about probability and statistics.
# MAGIC
# MAGIC x is a random variable, a scalar quantity, that measured N times.
# MAGIC
# MAGIC \\(x_i\\) is a single measurement with i = 1, ..., N
# MAGIC
# MAGIC {\\(x_i\\)} refers to the set of all N measurements
# MAGIC
# MAGIC We are generally trying to estimate \\(h(x)\\), the true distribution from which the values of \\(x\\) are drawn. We will refer to \\(h(x)\\) as the probability density (distribution) function or the "pdf" and is the propobability of a value lying between \\(x\\) and \\(x + dx\\). A histogram is an example of a pdf.
# MAGIC
# MAGIC While \\(h(x)\\) is the "true" distribution (or population pdf), what we measure from the data is the empirical distribution, which is denoted \\(f(x)\\),. So, \\(f(x)\\) is a model of \\(h(x)\\). From a frequentist perspective, given infinite data \\(f(x) \rightarrow h(x)\\), but in reality measurement errors keep this from being strictly true.
# MAGIC
# MAGIC If we are attempting to guess a model for \\(h(x)\\), then the process is parametric. With a model solution we can generate new (simulated) data that should mimic what we measure.
# MAGIC
# MAGIC If we are not attempting to guess a model, then the process is nonparametic. That is we are just trying to describe the data that we see in the most compact manner that we can, but we are not trying to produce mock data. The histograms that we made last time are an example of a nonparametric method of describing data.

# COMMAND ----------

# import general libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc

# import sklearn modules
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

# initialize seaborn to enhance matplotlib plots
sns.set()


# Set global font size
mpl.rcParams['font.size'] = 14


#------------------------------------------------------------
# Generate our data: a mix of several Cauchy distributions
# this is the same data used in the Bayesian Blocks figure
random_state = np.random.RandomState(seed=0)
N = 2000
mu_gamma_f = [(5, 1.0, 0.1),
              (7, 0.5, 0.5),
              (9, 0.1, 0.1),
              (12, 0.5, 0.2),
              (14, 1.0, 0.1)]
hx = lambda x: sum([f * sc.stats.cauchy(mu, gamma).pdf(x)
                    for (mu, gamma, f) in mu_gamma_f])
x = np.concatenate([sc.stats.cauchy(mu, gamma).rvs(int(f * N), random_state=random_state)
                    for (mu, gamma, f) in mu_gamma_f])
random_state.shuffle(x)
x = x[x > -10]
x = x[x < 30]


#------------------------------------------------------------
# plot the results
fig,ax = plt.subplots(figsize=(10, 10))

# create an evenly spaced number over a specified ranges
# in thsi case 1000 number evenly spaced between -10 and 30
xgrid = np.linspace(-10, 30, 1000)

# Compute density with KDE
kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
kde.fit(x[:, None])
dens_kde = np.exp(kde.score_samples(xgrid[:, None]))

# Compute density via Gaussian Mixtures using a pre-defined number of clusters (13)
gmm = GaussianMixture(n_components=13).fit(x.reshape(-1, 1))
logprob = gmm.score_samples(xgrid.reshape(-1, 1))
fx = lambda j : np.exp(gmm.score_samples(j.reshape(-1, 1)))

# plot the results
ax.plot(xgrid, 
        hx(xgrid), 
        ':', 
        color='black', 
        zorder=3,
        label="$h(x)$, Generating Distribution")

ax.plot(xgrid, 
        fx(np.array(xgrid)), 
        '-',
        color='gray',
        label="$f(x)$, parametric (13 Gaussians)")

ax.plot(xgrid, 
        dens_kde, 
        '-', 
        color='black', 
        zorder=3,
        label="$f(x)$, non-parametric (KDE)")

# label the plot
ax.text(0.02, 
        0.95, 
        s="%i points" % N, 
        ha='left', 
        va='top',
        transform=ax.transAxes)

ax.set_ylabel('$p(x)$')
ax.legend(loc='upper right')

ax.set_xlabel('$x$')
ax.set_xlim(0, 20)
ax.set_ylim(-0.01, 0.4001)

plt.show()

# COMMAND ----------


