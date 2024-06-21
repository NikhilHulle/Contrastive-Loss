import datasets as ds
from base_dataset import BaseDataset
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

class WinogroundDataset(BaseDataset):
    def __init__(self, split='test'):
        super().__init__("Winoground")
        load_dotenv()  
        self.token = os.getenv('HUGGINGFACE_TOKEN')
        self.dataset = None
        self.split = split  # Allow specifying which split to load (e.g., 'train', 'test')

    def load_dataset(self):
        # Load specified split of the dataset
        print(f"Loading {self.split} data...")
        self.dataset = ds.load_dataset('facebook/winoground', use_auth_token=self.token)[self.split]

    def process_dataset(self):
        # This method can be used to apply any preprocessing steps to the dataset
        pass

    def __getitem__(self, index):
        # Retrieve a single sample from the dataset
        sample = self.dataset[index]
        
        # Load image from URL
        response = requests.get(sample["image_url"])
        image = Image.open(BytesIO(response.content))
        
        caption = sample["caption"]
        label = sample["label"]
        
        return {
            "image": image,
            "caption": caption,
            "label": label
        }

    def __len__(self):
        return len(self.dataset)

    def inspect_dataset(self, index=0):
        # Pretty print a single dataset entry
        # sample = self.__getitem__(index)
        # print(f"Caption: {sample['caption']}")
        # print(f"Label: {sample['label']}")
        
        # # Display the image using matplotlib
        # plt.imshow(sample['image'])
        # plt.axis('off')  # Hide axes
        # plt.show()
        print(self.dataset[index])

