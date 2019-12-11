import numpy as np

from typing import Callable


class CEM():
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float = 0.0001):
        self.evaluationFunction = evaluationFunction
        self._name = "Cross_Entropy_Method"
        self._inital_theta = theta
        self._initial_sigma = sigma * np.identity(theta.shape[0])
        self._theta = theta  # TODO: set this value to the current mean parameter vector
        self._Sigma = self._initial_sigma  # TODO: set this value to the current covariance matrix
        self._popSize = popSize
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        # TODO
        return self._theta

    def train(self) -> np.ndarray:
        # TODO
        Temp_list = []
        for k in range(self._popSize):
            theta_k = np.random.multivariate_normal(self.parameters, self._Sigma)
            #print("theta_k",theta_k)
            j_k = self.evaluationFunction(theta_k)
            #print("Return:",j_k)
            Temp_list.append([theta_k, j_k])
        Temp_list.sort(key=lambda l: (l[1]), reverse=True)
        Temp_list_np = np.array(Temp_list)
        theta_k_population = Temp_list_np[:, 0]
        theta_elite = np.array(theta_k_population[: self.numElite])
        self._theta = np.mean(theta_k_population[: self.numElite])
        #print("calculated theta",self._theta)
        temp_Sigma = np.zeros(self._Sigma.shape)
        for i in range(self.numElite):
            temp_Sigma += np.dot((theta_elite[i] - self._theta)[:, np.newaxis],
                                 (theta_elite[i] - self._theta)[:, np.newaxis].T)
        self._Sigma = (temp_Sigma + self._epsilon * np.eye(self._Sigma.shape[0])) / (self._epsilon + self.numElite)

        return self.parameters

    def reset(self) -> None:
        # TODO
        self._theta = self._inital_theta
        self._Sigma = self._initial_sigma


# def evaluate_policy(theta, episodes):
#     G = np.zeros(episodes)
#     for i in range(episodes):
#         G[i] = 1
#     return np.mean(G)
