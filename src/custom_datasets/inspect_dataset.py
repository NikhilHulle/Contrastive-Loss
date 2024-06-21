from mscoco_dataset import MSCOCODataset
from winoground_dataset import WinogroundDataset

def main():
    dataset = MSCOCODataset(year=2017, decode_rle=True)
    dataset.load_dataset()
    dataset.inspect_dataset()

    for i in range(5):  # Check the first 5 entries
        data = dataset.__getitem__(i)
        print(data)


    # Assuming the dataset has been instantiated
    for i in range(5):
        print(dataset[i])  # Check the first 5 entries to ensure consistency

    # dataset = WinogroundDataset(split='test')
    # dataset.load_dataset()
    # dataset.inspect_dataset(0)

if __name__ == "__main__":
    main()