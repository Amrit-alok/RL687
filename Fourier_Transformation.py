import itertools

import numpy as np


class LinearSoftmax:

    def __init__(self, numActions: int, order, State_dims, sigma=1):
        self._order = order
        self._numActions = numActions
        self._sigma = sigma
        # self._theta = theta
        # self._action = action
        self.C = np.asarray(list(itertools.product(np.arange(order + 1), repeat=State_dims)))
        self.feature_size = (order + 1) ** State_dims
        self.fourier_base = np.zeros(self.feature_size)

    # @property
    # def parameters(self) -> np.ndarray:
    #     """
    #     Return the policy parameters as a numpy vector (1D array).
    #     This should be a vector of length |S|x|A|
    #     """
    #     return self._theta.flatten()
    #
    # @parameters.setter
    # def parameters(self, p: np.ndarray):
    #     """
    #     Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
    #     """
    #     self._theta = p.reshape(self._theta.shape)

    def fourier_features(self, state):
        value_func = np.cos(np.pi * np.dot(self.C, state))
        self.fourier_base = value_func.reshape((self.feature_size, 1))

    # def score(self):
    #     # print("Action in Fourier",self._action)
    #     theta = []
    #     if self._action == 0:
    #         theta = self._theta[:2]
    #     if self._action == 1:
    #         theta = self._theta[2:]
    #     print("theta", theta)
    #     return np.dot(theta, self.fourier_base)
    #     # return np.dot(self._theta, self.fourier_base)

    def linear_softmax_probs(self, theta, state, action):
        self.fourier_features(state)
        temp_theta = []
        if action == 0:
            temp_theta = theta[:2]
        if action == 1:
            temp_theta = theta[2:]
        # print("theta",theta)
        score = np.dot(temp_theta, self.fourier_base)
        numerator = np.exp(self._sigma * score)
        denominator = np.exp(np.dot(theta[:2], self.fourier_base)) + np.exp(
            np.dot(theta[2:], self.fourier_base))

        # #score = self.score()
        # score = np.asarray(score.flatten())
        #
        # exp_scores = np.exp(self._sigma * score)
        # softmax_probs = exp_scores / np.sum(exp_scores)
        # # print(softmax_probs)
        # # return softmax_probs.reshape(self._numActions)
        return np.divide(numerator, denominator)

    # def __call__(self, state: int, action=None) -> Union[float, np.ndarray]:
    #     # TODO
    #     if action != None:
    #         return self.linear_softmax_probs(state)[action]
    #     return self.linear_softmax_probs(state)
    #
    # def sampleAction(self, state: int) -> int:
    #
    #     return np.random.choice(np.arange(self._numActions), 1, p=self.linear_softmax_probs(state))[0]
