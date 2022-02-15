import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from scipy.special import gammaln, multigammaln, comb
from decorator import decorator
from numpy.linalg import inv
from scipy import stats
from itertools import islice
import collections
import torch
import gpytorch


try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.special import logsumexp
    # print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.sigmoid(self.linear(self.flatten(x)))


class DNN1(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_dim, output_size):
        super(DNN1, self).__init__()

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(input_size, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return F.sigmoid(self.layer(x))


class DNN2(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_dim, output_size):
        super(DNN2, self).__init__()

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(input_size, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim*3),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return self.layer(x)


class DNN3(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_dim, output_size):
        super(DNN3, self).__init__()

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return self.layer(x)


class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, seq_length, hidden_dim, input_size, output_size):
        super(CNN, self).__init__()

        # self.layer = nn.Sequential(
        #     nn.Conv1d(in_channels=seq_length, out_channels=output_size, kernel_size=input_size),
        #     nn.ReLU()
        # )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_dim, kernel_size=seq_length),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.layer2(x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_dimension):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_dimension
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_dimension, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class DeterministicEncoder(object):
#     """The Encoder."""
#
#     def __init__(self, output_sizes):
#         """CNP encoder.
#
#         Args:
#           output_sizes: An iterable containing the output sizes of the encoding MLP.
#         """
#         self._output_sizes = output_sizes
#
#     def __call__(self, context_x, context_y, num_context_points):
#         """Encodes the inputs into one representation.
#
#         Args:
#           context_x: Tensor of size bs x observations x m_ch. For this 1D regression
#               task this corresponds to the x-values.
#           context_y: Tensor of size bs x observations x d_ch. For this 1D regression
#               task this corresponds to the y-values.
#           num_context_points: A tensor containing a single scalar that indicates the
#               number of context_points provided in this iteration.
#
#         Returns:
#           representation: The encoded representation averaged over all context
#               points.
#         """
#
#         # Concatenate x and y along the filter axes
#         encoder_input = tf.concat([context_x, context_y], axis=-1)
#
#         # Get the shapes of the input and reshape to parallelise across observations
#         batch_size, _, filter_size = encoder_input.shape.as_list()
#         hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
#         hidden.set_shape((None, filter_size))
#
#         # Pass through MLP
#         with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
#             for i, size in enumerate(self._output_sizes[:-1]):
#                 hidden = tf.nn.relu(tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))
#
#                 # Last layer without a ReLu
#                 hidden = tf.layers.dense(hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))
#
#         # Bring back into original shape
#         hidden = tf.reshape(hidden, (batch_size, num_context_points, size))
#
#         # Aggregator: take the mean over all points
#         representation = tf.reduce_mean(hidden, axis=1)
#
#         return representation
#
#
# class DeterministicDecoder(object):
#     """The Decoder."""
#
#     def __init__(self, output_sizes):
#         """CNP decoder.
#
#         Args:
#           output_sizes: An iterable containing the output sizes of the decoder MLP
#               as defined in `basic.Linear`.
#         """
#         self._output_sizes = output_sizes
#
#     def __call__(self, representation, target_x, num_total_points):
#         """Decodes the individual targets.
#
#         Args:
#           representation: The encoded representation of the context
#           target_x: The x locations for the target query
#           num_total_points: The number of target points.
#
#         Returns:
#           dist: A multivariate Gaussian over the target points.
#           mu: The mean of the multivariate Gaussian.
#           sigma: The standard deviation of the multivariate Gaussian.
#         """
#
#         # Concatenate the representation and the target_x
#         representation = tf.tile(
#             tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
#         input = tf.concat([representation, target_x], axis=-1)
#
#         # Get the shapes of the input and reshape to parallelise across observations
#         batch_size, _, filter_size = input.shape.as_list()
#         hidden = tf.reshape(input, (batch_size * num_total_points, -1))
#         hidden.set_shape((None, filter_size))
#
#         # Pass through MLP
#         with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
#             for i, size in enumerate(self._output_sizes[:-1]):
#                 hidden = tf.nn.relu(
#                     tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))
#
#           # Last layer without a ReLu
#             hidden = tf.layers.dense(hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))
#
#         # Bring back into original shape
#         hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))
#
#         # Get the mean an the variance
#         mu, log_sigma = tf.split(hidden, 2, axis=-1)
#
#         # Bound the variance
#         sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
#
#         # Get the distribution
#         dist = tf.contrib.distributions.MultivariateNormalDiag(
#             loc=mu, scale_diag=sigma)
#
#         return dist, mu, sigma
#
#
# class DeterministicModel(object):
#     """The CNP model."""
#
#     def __init__(self, encoder_output_sizes, decoder_output_sizes):
#         """Initialises the model.
#
#         Args:
#           encoder_output_sizes: An iterable containing the sizes of hidden layers of
#               the encoder. The last one is the size of the representation r.
#           decoder_output_sizes: An iterable containing the sizes of hidden layers of
#               the decoder. The last element should correspond to the dimension of
#               the y * 2 (it encodes both mean and variance concatenated)
#         """
#         self._encoder = DeterministicEncoder(encoder_output_sizes)
#         self._decoder = DeterministicDecoder(decoder_output_sizes)
#
#     def __call__(self, query, num_total_points, num_contexts, target_y=None):
#         """Returns the predicted mean and variance at the target points.
#
#         Args:
#           query: Array containing ((context_x, context_y), target_x) where:
#               context_x: Array of shape batch_size x num_context x 1 contains the
#                   x values of the context points.
#               context_y: Array of shape batch_size x num_context x 1 contains the
#                   y values of the context points.
#               target_x: Array of shape batch_size x num_target x 1 contains the
#                   x values of the target points.
#           target_y: The ground truth y values of the target y. An array of
#               shape batchsize x num_targets x 1.
#           num_total_points: Number of target points.
#
#         Returns:
#           log_p: The log_probability of the target_y given the predicted
#           distribution.
#           mu: The mean of the predicted distribution.
#           sigma: The variance of the predicted distribution.
#         """
#
#         (context_x, context_y), target_x = query
#
#         # Pass query through the encoder and the decoder
#         representation = self._encoder(context_x, context_y, num_contexts)
#         dist, mu, sigma = self._decoder(representation, target_x, num_total_points)
#
#         # If we want to calculate the log_prob for training we will make use of the
#         # target_y. At test time the target_y is not available so we return None
#         if target_y is not None:
#             log_p = dist.log_prob(target_y)
#         else:
#             log_p = None
#
#         return log_p, mu, sigma


def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[0]

    if not np.array_equal(f.data, args[0]):
        f.cache = {}
        f.data = args[0]

    try:
        f.cache[args[1:3]]
    except KeyError:
        f.cache[args[1:3]] = f(*args, **kwargs)
    return f.cache[args[1:3]]


def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)


def offline_changepoint_detection(data, prior_func, observation_log_likelihood_function, truncate=-np.inf):
    """Compute the likelihood of change points on data.

    Keyword arguments:
    data                                -- the time series data
    prior_func                          -- a function yields the likelihood of a change point given the distance to the last data point (at n)
    observation_log_likelihood_function -- a function giving the log likelihood of a data part
    truncate                            -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

    P                                   -- the likelihoods if pre-computed
    """

    n = len(data)  # The number of time points
    # A prior of how probable it is to have two successive change points with the distance t.
    # Assume a uniform prior over the length of sequences (const_prior)
    g = np.zeros((n,))
    Q = np.zeros((n,))  # The log-likelihood of data [t, n]
    G = np.zeros((n,))
    P = np.ones((n, n)) * -np.inf  # The log-likelihood of a data sequence [t, s], given there is no change point between t and s
    Pcp = np.ones((n - 1, n - 1)) * -np.inf
    # The probability of CP at each time point = The log-likelihood that the i-th change point is at time step t

    # save everything in log representation
    for t in range(n):
        g[t] = np.log(prior_func(t))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])
        # print("prior_function({}) ~>".format(t), prior_func(t), "G({}) ~>  ".format(t), G[t])

    # print "\n", "G ~~~~>", G

    P[n-1, n-1] = observation_log_likelihood_function(data, n-1, n)
    Q[n-1] = P[n-1, n-1]

    for t in reversed(range(n-1)):
        if t % 500 == 0:
            print("t  ~~~>", t, "out of total n ==", n)
        P_next_cp = -np.inf  # == log(0)
        # print range(t, n-1)
        for s in range(t, n-1):
            # print("s  ~~~>", s, "out of total n ~~~~>", range(t, n-1))
            P[t, s] = observation_log_likelihood_function(data, t, s+1)
            # print "(t, s) ~~~> ", (t, s), "s+1 ~~>", s+1, "s+1-t ~~~>", s+1-t
            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see Fearnhead, 2006, eq. (3))

            if summand - P_next_cp < truncate:
                break
        # print "(t, n-1) ~~~> ", (t, n-1)
        P[t, n-1] = observation_log_likelihood_function(data, t, n)

        # print "n-1-t ~~> ", n-1-t, "~~>", G[n-1-t]
        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t]))
            # print("antiG ~~~>", antiG)
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])
            # print("antiG ~~~>", antiG)

        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

    for t in range(n-1):
        # print("t  ~~~>", t, "out of total n-2 ~~~~>", n-2)
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    for j in range(1, n-1):
        for t in range(j, n-1):
            tmp_cond = Pcp[j-1, j-1:t] + P[j:t+1, t] + Q[t + 1] + g[0:t-j+1] - Q[j:t+1]
            Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf
    # print("P:", P, "\n", "Q:", Q, "\n", "Pcp:", np.exp(Pcp), np.exp(Pcp).sum(0))
    return Q, P, Pcp


@dynamic_programming
def gaussian_obs_log_likelihood(data, t, s):
    s += 1
    n = s - t
    mean = data[t:s].sum(0) / n

    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = 1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)

    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (data[t:s] - muT)**2/(nuT * scale)))
    lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)

    return np.sum(n * lgA - (nuT + 1)/2 * prob)


def ifm_obs_log_likelihood(data, t, s):
    '''Independent Features model from xuan et al'''
    s += 1
    n = s - t
    x = data[t:s]
    if len(x.shape)==2:
        d = x.shape[1]
    else:
        d = 1
        x = np.atleast_2d(x).T

    N0 = d          # weakest prior we can use to retain proper prior
    V0 = np.var(x)
    Vn = V0 + (x**2).sum(0)

    # sum over dimension and return (section 3.1 from Xuan paper):
    return d*(-(n/2)*np.log(np.pi) + (N0/2)*np.log(V0) - gammaln(N0/2) + gammaln((N0+n)/2)) - \
        (((N0+n)/2)*np.log(Vn)).sum(0)


def fullcov_obs_log_likelihood(data, t, s):
    s += 1
    n = s - t
    # print "s:", s, "t:", t
    x = data[t:s]
    # print x.shape

    if len(x.shape) == 2:
        dim = x.shape[1]
    else:
        dim = 1
        x = np.atleast_2d(x).T

    N0 = dim          # weakest prior we can use to retain proper prior
    # print dim, x, np.var(x)
    V0 = np.var(x)*np.eye(dim)
    # print x, "\n", np.var(x), "\n", V0

    # Improvement over np.outer
    # http://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
    # Vn = V0 + np.array([np.outer(x[i], x[i].T) for i in xrange(x.shape[0])]).sum(0)
    Vn = V0 + np.einsum('ij,ik->jk', x, x)

    # section 3.2 from Xuan paper:
    return -(dim*n/2)*np.log(np.pi) + (N0/2)*np.linalg.slogdet(V0)[1] - \
        multigammaln(N0/2, dim) + multigammaln((N0+n)/2, dim) - ((N0+n)/2)*np.linalg.slogdet(Vn)[1]


def const_prior(r, l):
    return 1/l


def geometric_prior(t, p):
    return p * ((1 - p) ** (t - 1))


def neg_binominal_prior(t, k, p):
    return comb(t - k, k - 1) * p ** k * (1 - p) ** (t - k)


def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum(R[0:t+1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])

        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

        maxes[t] = R[:, t].argmax()
    return R, maxes


class BOCD(object):
    def __init__(self, hazard_function, observation_likelihood):
        """Initializes th detector with zero observations.
        """
        self.t0 = 0
        self.t = -1
        self.growth_probs = np.array([1.])
        self.hazard_function = hazard_function
        self.observation_likelihood = observation_likelihood

    def update(self, x):
        """Updates change point probabilities with a new data point.
        """
        self.t += 1

        t = self.t - self.t0

        # allocate enough space
        if len(self.growth_probs) == t + 1:
            self.growth_probs = np.resize(self.growth_probs, (t + 1) * 2)
            # print("resize posterior probability:", self.growth_probs)

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = self.observation_likelihood.pdf(x)
        # print("pred_probs:", pred_probs)

        # Evaluate the hazard function for this interval
        H = self.hazard_function(np.array(range(t + 1)))

        # Evaluate the probability that there *was* a change point and we're
        # accumulating the mass back down at r = 0.
        cp_prob = np.sum(self.growth_probs[0:t + 1] * pred_probs * H)

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive probabilities.
        self.growth_probs[1:t + 2] = self.growth_probs[0:t + 1] * pred_probs * (1-H)
        # Put back change point probability
        self.growth_probs[0] = cp_prob

        # Renormalize the run length probabilities for improved numerical stability.
        self.growth_probs[0:t + 2] = self.growth_probs[0:t + 2] / np.sum(self.growth_probs[0:t + 2])

        # Update the parameter sets for each possible run length.
        self.observation_likelihood.update_theta(x)

    def prune(self, t0):
        """prunes memory before time t0. That is, pruning at t=0
        does not change the memory. One should prune at times
        which are likely to correspond to changepoints.
        """
        self.t0 = t0
        self.observation_likelihood.prune(self.t - t0 + 1)


def constant_hazard(lam, r):
    """
    Computes the "constant" hazard, that is corresponding to Poisson process.
    """
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data,
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha * self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data - self.mu)**2) / (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

    def prune(self, t):
        """Prunes memory before t.
        """
        self.mu = self.mu[:t + 1]
        self.kappa = self.kappa[:t + 1]
        self.alpha = self.alpha[:t + 1]
        self.beta = self.beta[:t + 1]


class MultivariateT:
    def __init__(self, dims, dof=None, kappa=1, mu=None, scale=None):
        """
        Create a new predictor using the multivariate student T distribution as the posterior predictive.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
             Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: The number of observations we've already seen
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof is None:
            dof = dims + 1
        # The default mean is all 0s
        if mu is None:
            mu = [0]*dims
        # The default covariance is the identity matrix. The scale is the inverse of that, which is also the identity
        if scale is None:
            scale = np.identity(dims)

        # Track time
        self.t = 0

        # The dimensionality of the dataset (number of variables)
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data):
        """
        Returns the probability of the observed data under the current and historical parameters
        :param data: A 1 x D vector of new data
        """
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(enumerate(zip(
                t_dof,
                self.mu,
                inv(expanded * self.scale)
            )), self.t):
                ret[i] = stats.multivariate_t.pdf(
                    x=data,
                    df=df,
                    loc=loc,
                    shape=shape
                )
        except AttributeError:
            raise Exception('You need scipy 1.6.0 or greater to use the multivariate t distribution')
        return ret

    def update_theta(self, data):
        """
        Performs a bayesian update on the prior parameters, given data
        :param data: A 1 x D vector of new data
        """
        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate([
            self.scale[:1],
            inv(
                inv(self.scale)
                + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2)) * (np.matmul(np.expand_dims(centered, 2), np.expand_dims(centered, 1)))
            )
        ])
        self.mu = np.concatenate([self.mu[:1], (np.expand_dims(self.kappa, 1) * self.mu + data)/np.expand_dims(self.kappa + 1, 1)])
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])


# Plots the run length distributions along with a dataset
def plot(R, data):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(data)
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    sparsity = 1  # only plot every fifth data for faster display
    ax.pcolor(
        np.array(range(0, len(R[:, 0]), sparsity)),
        np.array(range(0, len(R[:, 0]), sparsity)),
        np.log(R),
        cmap=cm.Greys, vmin=-30, vmax=0
    )
    return fig