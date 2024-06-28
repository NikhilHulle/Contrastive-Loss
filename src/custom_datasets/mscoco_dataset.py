# # from datasets import load_dataset
from datasets import load_dataset as hf_load_dataset
from .base_dataset import BaseDataset
import pyarrow.parquet as pq
from collections import defaultdict
import json
import matplotlib.pyplot as plt


class MSCOCODataset(BaseDataset):
    def __init__(self, year=2017, coco_task='instances',split = 'train', decode_rle=True, transform=None, tokenizer = None):
        super().__init__(coco_task)
        self.year = year
        self.decode_rle = decode_rle
        self.dataset = None
        self.coco_task = coco_task
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.load_dataset()
        # self.filter_dataset_by_split()
        # self.dataset = self.dataset.select(range(70))

    def load_dataset(self):

        

        self.dataset = hf_load_dataset(
            "shunk031/MSCOCO",
            year=2014,
            coco_task="captions",
            trust_remote_code=True,
        )
        print("Available splits:", list(self.dataset.keys()))
        print("Sample entry from the train split:", self.dataset['train'][0])
        # full_dataset = hf_load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
        # self.dataset = full_dataset[self.split]  # Access the specific split
        


    def process_dataset(self):
        pass

    # def get_sentence_by_sentid(self, sentid):
    #     # Iterate over the dataset to find the sentence with the given sentid
    #     for entry in self.dataset[self.split]:
    #         if 'sentences' in entry and entry['sentences']['sentid'] == sentid:
    #             return entry['sentences']
    #     return None  
    
    # def filter_dataset_by_split(self):
    #     # Filter the dataset entries by the 'split' column
    #     self.dataset = self.dataset.filter(lambda x: x['split'] == self.split)


    def __getitem__(self, index):
        item = self.dataset[self.split][index]
        image = item['image']
        annotations = item['annotations']

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply image transformations
        if self.transform is not None:
            image = self.transform(image)

        captions = annotations['caption']

        #if self.tokenizer:
            #captions = self.tokenizer(captions, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
        
        if self.tokenizer:
        # Tokenize all captions
          encoded_captions = self.tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
          return {'image': image, 'captions': encoded_captions['input_ids']}
    
        return {'image': image, 'captions': captions}

        # print(f"Item keys: {item.keys()}")
        # return {'image': image, 'captions': captions}

        #return {'image': image, 'captions': captions['input_ids'].squeeze()}

    def __len__(self):
        return len(self.dataset[self.split])

    def inspect_dataset(self, index=0):
    # Pretty print a single dataset entry
        item = self.dataset[self.split][index]  # Access the dataset using the specified split
        print(item['annotations']['caption'])
        image = item['image']
        plt.imshow(image)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

        


