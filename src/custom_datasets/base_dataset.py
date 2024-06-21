from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = None

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def process_dataset(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass