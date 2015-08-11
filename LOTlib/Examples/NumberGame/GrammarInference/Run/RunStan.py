__author__ = 'Eric Bigelow'


# ============================================================================================================
# Stan code
# =========

stan_code = """
data {
    int<lower=1> h;                     // # of domain hypotheses
    int<lower=1> p;                     // # of rules proposed to
    int<lower=1> r;                     // # of rules in total
    int<lower=0> d;                     // # of data points
    int<lower=0> q;                     // max # of queries for a given datum

    real<lower=0> x_init[r];            // Initial rule probabilites (used for non-proposed indexes)
    int<lower=0> P[p];                  // proposal indexes
    matrix<lower=0>[h,r] C;             // rule counts for each hypothesis
    vector<upper=0>[h] L[d];            // log likelihood of data.input
    vector<lower=0,upper=1>[h] R[d,q];  // is each data.query in each hypothesis  (1/0)
    real<lower=0> D[d,q,2];             // human response for each data.query  (# yes, # no)

    real alpha;                         // shape of prior gamma
    real beta;                          // inverse scale of prior gamma
}

parameters {
    real<lower=0> x_propose[p];         // normalized vector of rule probabilities
}

transformed parameters {
    real<lower=0> x_full[r];

    x_full <- x_init
    for (i in 1:p) {
        x_full[P[i]] <- x_propose[P[i]];
    }
}

model {
    vector[h] priors;
    vector[h] posteriors;
    vector[h] w;
    real Z;
    real k;
    real n;
    real pr;
    real bc;

    // Prior
    increment_log_prob(gamma_log(x_full,alpha,beta));       // TODO: should alpha / beta be vectors?

    // Likelihood model
    priors <- C * x_full;                   // prior for each hypothesis

    for (i in 1:d) {
        posteriors <- L[i] + priors;
        Z <- log_sum_exp(posteriors);
        w <- exp(posteriors - Z);           // weights for each hypothesis

        for (j in 1:q) {
            k <- D[i,j,1];                  // num. yes responses
            n <- D[i,j,1] + D[i,j,2];       // num. trials

            // If we have human responses for this query
            if (n > 0) {
                pr <- log(sum(w .* R[i, j]));                           // logsum of binary values for yes/no
                bc <- tgamma(n+1) - (tgamma(k+1) + tgamma(n-k+1));      // binomial coefficient
                increment_log_prob(bc + (k*pr) + (n-k)*log1m_exp(pr));  // likelihood we got human output
            }
        }
    }

}
"""


# ============================================================================================================
# Fitting the model in pystan!
# ============================
import pystan, pickle
import numpy as np


date_dir = '8_11/'

with open(date_dir+'stan_data.p', 'rb') as f:
    stan_data = pickle.load(f)

stan_model = pystan.StanModel(model_code=stan_code)
fit = stan_model.sampling(data=stan_data, iter=1000, chains=4, warmup=10)

print(fit)
with open(date_dir+'stan_model.p', 'w') as f:
    pickle.dump(stan_model, f)
with open(date_dir+'stan_fit.p', 'w') as f:
    pickle.dump(fit, f)

la = fit.extract(permuted=True)  # return a dictionary of arrays
mu = la['mu']

# return an array of three dimensions: iterations, chains, parameters
a = fit.extract(permuted=False)
# fit.plot()















# ============================================================================================================
# X: pystan example code
# ======================
#
# schools_code = """
# data {
#     int<lower=0> J; // number of schools
#     real y[J]; // estimated treatment effects
#     real<lower=0> sigma[J]; // s.e. of effect estimates
# }
# parameters {
#     real mu;
#     real<lower=0> tau;
#     real eta[J];
# }
# transformed parameters {
#     real theta[J];
#     for (j in 1:J)
#     theta[j] <- mu + tau * eta[j];
# }
# model {
#     eta ~ normal(0, 1);
#     y ~ normal(theta, sigma);
# }
# """
#
# schools_dat = {'J': 8,
#                'y': [28,  8, -3,  7, -1,  1, 18, 12],
#                'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}
#
# fit = pystan.stan(model_code=schools_code, data=schools_dat,
#                   iter=1000, chains=4)

