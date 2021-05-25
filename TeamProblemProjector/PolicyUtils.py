import numpy as np


# class AlphaVectorsPolicy:
#     def __init__(self, alpha_vectors, actions, gamma):
#         assert alpha_vectors.shape[0] == len(actions)
#         self.alpha_vectors = alpha_vectors
#         self.actions = actions
#         self.gamma = gamma
#         self.num_states = alpha_vectors.shape[1]
#         self.reward_function = None
#         self.transition_function = None
#
#     def Q(self, s):
#         return self.R(s, a) + self.gamma * self.V()
#
#     def V(self, s):
#         pass
#
#     def P(self, b=None, s=None):
#         if b is not None and s is None:
#             b = np.zeros(self.num_states)
#             b[s] = 1
#         self.actions[np.argmax(np.dot(b, self.alpha_vectors))
