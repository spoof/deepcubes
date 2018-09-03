from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def loo_accuracy(self, X, Y):
        pass
