from abc import ABC, abstractmethod


class ProblemConstants(ABC):
    @staticmethod
    @abstractmethod
    def is_public_action(action):
        pass

    @staticmethod
    @abstractmethod
    def is_sense_action(action):
        pass

    @staticmethod
    @abstractmethod
    def is_idle_action(action):
        pass

    @staticmethod
    @abstractmethod
    def is_collaborative_action(action):
        pass

    @staticmethod
    @abstractmethod
    def extract_agents_from_action(action):
        pass

    @staticmethod
    @abstractmethod
    def extract_objectives_from_action(action, metadata=None):
        pass

    @staticmethod
    @abstractmethod
    def extract_agents_from_obsvar(obsvar):
        pass

    @staticmethod
    @abstractmethod
    def extract_preconditions_from_action(action, metadata=None):
        pass

    @staticmethod
    @abstractmethod
    def extract_subjects_from_action(action, metadata=None):
        pass
