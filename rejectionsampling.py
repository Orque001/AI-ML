"""
b. Write a program to perform rejection sampling to estimate P(F = 1 | B = 1) and
P(F = 1 | C = 1). Perform separate runs of rejection sampling using N = 10, 100, 1000, 10000
samples. (Note that N refers to the total number of samples on a given run, many of which will be
rejected.) Show the results of your sampling runs by producing a plot of the estimated value for each
of the two probabilities as a function of N. Include the true value of each probability (computed in
part (a)) in the plot as a horizontal line.


c. Write a program to perform likelihood weighting to estimate P(F = 1 | B = 1) and
P(F = 1 | C = 1). Perform separate runs of likelihood weighting using N = 10, 100, 1000, 10000
samples. Show the results of your sampling runs by producing a plot of the estimated value for each of
the two probabilities as a function of N. Include the true value of each probability (computed in part
(a)) in the plot as a horizontal line
"""
import numpy as np
import matplotlib.pyplot as plt

# Probabilities
P_F1 = 0.01
P_E1 = 0.02
P_A1_F1_E1 = 0.95
P_A1_F1_E0 = 0.94
P_A1_F0_E1 = 0.29
P_A1_F0_E0 = 0.01
P_B1_A1 = 0.90
P_B1_A0 = 0.05
P_C1_A1 = 0.70
P_C1_A0 = 0.01

# generates sameples using the probabilities
def generate_sample():
    F = np.random.rand() < P_F1
    E = np.random.rand() < P_E1
    A = np.random.rand() < (P_A1_F1_E1 if F and E else
                            P_A1_F1_E0 if F and not E else
                            P_A1_F0_E1 if not F and E else
                            P_A1_F0_E0)
    B = np.random.rand() < (P_B1_A1 if A else P_B1_A0)
    C = np.random.rand() < (P_C1_A1 if A else P_C1_A0)
    return F, E, A, B, C

# rejection_sampling() function to estimate the conditional probability
def rejection_sampling(N, evidence_var):
    accepted_F = 0
    accepted_samples = 0
    for _ in range(N):
        F, _, _, B, C = generate_sample()
        if (evidence_var == 'B' and B) or (evidence_var == 'C' and C):
            accepted_samples += 1
            if F:
                accepted_F += 1
    return (accepted_F / accepted_samples) if accepted_samples > 0 else 0

# likelihood_weighting() function to estimate the conditional probability
def likelihood_weighting(N, evidence_var):
    weighted_F = 0
    total_weight = 0
    for _ in range(N):
        F, _, A, B, C = generate_sample()
        weight = P_B1_A1 if (evidence_var == 'B' and B) else \
                 P_B1_A0 if (evidence_var == 'B' and not B) else \
                 P_C1_A1 if (evidence_var == 'C' and C) else \
                 P_C1_A0
        total_weight += weight
        if F:
            weighted_F += weight
    return (weighted_F / total_weight) if total_weight > 0 else 0

# data structures to store results
sample_sizes = [10, 100, 1000, 10000]
results_rejection_B = []
results_rejection_C = []
results_likelihood_B = []
results_likelihood_C = []

# add results into the lists by calling rejection_sampling and likelihood_weighting functions
for N in sample_sizes:
    results_rejection_B.append(rejection_sampling(N, 'B'))
    results_rejection_C.append(rejection_sampling(N, 'C'))
    results_likelihood_B.append(likelihood_weighting(N, 'B'))
    results_likelihood_C.append(likelihood_weighting(N, 'C'))

# plot and display both Rejection Sampling and Likelihood Weighting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, results_rejection_B, label='Rejection P(F=1 | B=1)', marker='o')
plt.plot(sample_sizes, results_rejection_C, label='Rejection P(F=1 | C=1)', marker='o')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated Probability')
plt.xscale('log')
plt.title('Rejection Sampling Estimates')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, results_likelihood_B, label='Likelihood P(F=1 | B=1)', marker='o')
plt.plot(sample_sizes, results_likelihood_C, label='Likelihood P(F=1 | C=1)', marker='o')
plt.xlabel('Number of Samples')
plt.xscale('log')
plt.title('Likelihood Weighting Estimates')
plt.legend()
plt.grid(True)

plt.show()
