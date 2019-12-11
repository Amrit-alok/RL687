import itertools
import numpy as np

from tabular_softmax import TabularSoftmax


class FourierTransformation:
    def __init__(self, theta, numActions, num_state_feature, order, action, sigma=1):
        self.theta = theta
        self.sigma = sigma
        self.action = action
        self.order = order
        self.num_state_feature = num_state_feature
        self.numActions = numActions
        self.state_feature_vector = np.asarray(
            list(itertools.product(np.arange(self.order + 1), repeat=self.num_state_feature)))

    def transformation(self, state):
        phi_state = np.cos(np.pi * self.state_feature_vector.dot(state))
        theta = self.theta.reshape(phi_state.shape[0], -1).T.dot(phi_state)
        tabular_policy = TabularSoftmax(numStates=1, numActions=self.numActions)
        tabular_policy.parameters = theta
        return tabular_policy.getActionProbabilities(0)
