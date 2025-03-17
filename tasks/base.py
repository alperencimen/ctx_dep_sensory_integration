from abc import ABC, abstractmethod

class Task(ABC):
    def __init__(self, root:str, version: str, duration: float, delta_t: float, num_trials: int, *args, **kwargs):
        self.root = root
        self.version = str.lower(version)
        self.duration = duration
        self.delta_t = delta_t
        self.num_trials = num_trials
        self.data = self._generate_dataset()
    
    @abstractmethod
    def _generate_dataset(self):
        """Generate the dataset for the task."""
        pass
    
    @abstractmethod
    def _check_input_validity(self):
        """ Check if the input values are valid. """
        pass
    
    @abstractmethod
    def _discretize_input(self):
        """ Discretize the input here """
        pass
    
    @abstractmethod
    def visualize_task(self):
        """Visualize the task with a plot"""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int):
        """Get the item at the given index.
        Args:
            index (int): index of the item to retrieve
        """
        pass
    
# Always add if __name__ == "main" block to visualize the task