import numpy as np
from typing import Union


class TabularSoftmax():
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates: int, numActions: int):
        # The internal policy parameters must be stored as a matrix of size
        # (numStates x numActions)
        # TODO
        self._theta = np.zeros((numStates, numActions))
        self.sigma = 1
        self._numActions = numActions

    @property
    def parameters(self) -> np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p: np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state: int, action=None) -> Union[float, np.ndarray]:
        # TODO

        #return Union[self.samplAction(state), self.getActionProbabilities(state)]
        if action is None:
            return self.getActionProbabilities(state)
        else:
            return self.getActionProbabilities(state)[action]

    def samplAction(self, state: int) -> int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        # TODO
        action_dist = self.getActionProbabilities(state)
        return np.random.choice(np.arange(self._numActions), 1, p=action_dist)[0]

    def getActionProbabilities(self, state: int) -> np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        # TODO
        # expo = np.exp(self.sigma * self._theta[state])
        # expo_sum = np.sum(expo)
        # return expo / expo_sum
        theta_s = self._theta[state, :] - np.max(self._theta[state, :])
        theta_s = np.exp(theta_s)
        theta_s /= np.sum(theta_s)

        return theta_s
