# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from torchvision import transforms
# from transformers import AutoTokenizer
# from custom_datasets.mscoco_dataset import MSCOCODataset
# from models.transformers import VisionTransformer, TextTransformer

# # def custom_collate_fn(batch):
# #     # Stack images and pad captions
# #     images = torch.stack([item['image'] for item in batch if item['image'] is not None])
# #     captions = pad_sequence([caption for item in batch for caption in item['captions']], batch_first=True, padding_value=0)
# #     return {'images': images, 'captions': captions}

# def custom_collate_fn(batch):    
#     images = torch.stack([item['image'] for item in batch if item['image'] is not None])
#     all_captions = [item['captions'] for item in batch]
    
    
#     captions_per_image = [len(captions) for captions in all_captions]
    
    
#     flattened_captions = [caption for sublist in all_captions for caption in sublist]
#     padded_captions = pad_sequence(flattened_captions, batch_first=True, padding_value=0)
    
    
#     print(f"Captions per image in this batch: {captions_per_image}")
    
#     return {'images': images, 'captions': padded_captions, 'captions_per_image': captions_per_image}

# def train(num_epochs, dataloader, model_vision, model_text, criterion, optimizer, tokenizer, device):
#     model_vision.train()
#     model_text.train()

#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch_idx, batch in enumerate(dataloader):
#             images, captions = batch['images'].to(device), batch['captions'].to(device)

#             print(f"Batch {batch_idx}: Images shape: {images.shape}, Captions count: {len(captions)}")  # Check batch shapes
            
#             # Move the images tensor to the device
#             # images = images.to(device)
            
#             # # Flatten the list of captions
#             # captions = [caption for sublist in captions for caption in sublist]
            
#             # # Tokenize the captions
#             # captions = tokenizer(captions, padding=True, truncation=True, return_tensors='pt').to(device)
            
            
            
#             # Forward pass
#             image_embeddings = model_vision(images)
#             text_embeddings = model_text(captions)

#             print(f"Image embeddings shape: {image_embeddings.shape}")
#             print(f"Text embeddings shape: {text_embeddings.shape}")
            
#             # Normalize embeddings to compute cosine similarity
#             image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
#             text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

#             # Compute cosine similarity as dot product of normalized embeddings
#             logits = torch.matmul(image_embeddings, text_embeddings.t())
#             labels = torch.arange(images.size(0)).to(device)
#             loss = criterion(logits, labels)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         epoch_loss /= len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
#     return model_vision, model_text


# def evaluate(model_vision, model_text, dataloader, device):
#     model_vision.eval()
#     model_text.eval()
    
#     total_loss = 0.0
#     correct_predictions = 0
#     total_samples = 0
    
#     criterion = nn.CrossEntropyLoss()
    
#     with torch.no_grad():
#         for batch in dataloader:
#             images, captions = batch['images'].to(device), batch['captions'].to(device)
            
#             # Forward pass
#             image_embeddings = model_vision(images)
#             text_embeddings = model_text(captions)
            
#             # Normalize embeddings to compute cosine similarity
#             image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
#             text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
            
#             # Compute cosine similarity as dot product of normalized embeddings
#             similarity_scores = torch.matmul(image_embeddings, text_embeddings.t())
            
#             # Compute loss
#             labels = torch.arange(images.size(0)).to(device)
#             loss = criterion(similarity_scores, labels)
            
#             total_loss += loss.item()
            
#             # Compute accuracy
#             _, predicted_labels = torch.max(similarity_scores, dim=1)
#             correct_predictions += (predicted_labels == labels).sum().item()
#             total_samples += len(labels)
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = correct_predictions / total_samples
    
#     return avg_loss, accuracy


# def main():
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Define the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    
    
#     # Define the data transformation
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Load the dataset
#     train_dataset = MSCOCODataset(split='train', transform=transform, tokenizer=tokenizer)
#     val_dataset = MSCOCODataset(split='validation', transform=transform, tokenizer=tokenizer)
    
    
    
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

#     for batch_idx, batch in enumerate(train_dataloader):
#         print(f"Batch {batch_idx} - Captions per image: {batch['captions_per_image']}")
        
#         # Optionally break after a few batches
#         if batch_idx >= 5:
#             break

#     print("Number of batches:", len(train_dataloader))

#     for batch in train_dataloader:
#         print("Batch Images Shape:", batch['images'].shape)
#         print("Batch Captions:", batch['captions'])
#         break 
    
#     # Create model instances
#     vision_transformer = VisionTransformer().to(device)
#     text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=64).to(device)
    
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(list(vision_transformer.parameters()) + list(text_transformer.parameters()))
    
#     # Training loop
#     num_epochs = 10
#     vision_transformer, text_transformer = train(num_epochs, train_dataloader, vision_transformer, text_transformer, criterion, optimizer, tokenizer, device)
    
#     # Evaluation
#     val_loss, val_accuracy = evaluate(vision_transformer, text_transformer, val_dataloader, tokenizer, transform, device)
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
#     # Save the trained models
#     torch.save(vision_transformer.state_dict(), 'vision_transformer.pth')
#     torch.save(text_transformer.state_dict(), 'text_transformer.pth')

# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import AutoTokenizer, DistilBertTokenizerFast
from custom_datasets.mscoco_dataset import MSCOCODataset
from models.transformers import VisionTransformer, TextTransformer
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch if item['image'] is not None])
    all_captions = [item['captions'] for item in batch]
    
    captions_per_image = [len(captions) for captions in all_captions]
    
    flattened_captions = [caption for sublist in all_captions for caption in sublist]
    padded_captions = pad_sequence(flattened_captions, batch_first=True, padding_value=0)
    
    return {'images': images, 'captions': padded_captions, 'captions_per_image': captions_per_image}

def diagonal_cross_entropy_loss(similarity_matrix, captions_per_image):
    batch_size = similarity_matrix.size(0)
    total_captions = similarity_matrix.size(1)
    
    # Create a mask for positive pairs
    pos_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    start_idx = 0
    for i, num_captions in enumerate(captions_per_image):
        pos_mask[i, start_idx:start_idx+num_captions] = True
        start_idx += num_captions
    
    # Compute log probabilities
    log_softmax = F.log_softmax(similarity_matrix, dim=1)
    
    # Compute positive pair loss
    pos_loss = -log_softmax[pos_mask].mean()
    
    # Compute negative pair loss (implicitly done by log_softmax)
    neg_loss = -log_softmax[~pos_mask].mean()
    
    # Total loss is the sum of positive and negative losses
    total_loss = pos_loss + neg_loss
    
    return total_loss

def compute_accuracy(similarity_matrix, captions_per_image):
    batch_size = similarity_matrix.size(0)
    total_captions = similarity_matrix.size(1)
    
    start_idx = 0
    correct = 0
    total = 0
    for i, num_captions in enumerate(captions_per_image):
        scores = similarity_matrix[i, start_idx:start_idx+num_captions]
        _, predicted = scores.max(0)
        correct += (predicted == 0).sum().item()
        total += num_captions
        start_idx += num_captions
    
    return correct / total

def train(num_epochs, dataloader, model_vision, model_text, optimizer, device):
    model_vision.train()
    model_text.train()

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images, captions = batch['images'].to(device), batch['captions'].to(device)
            captions_per_image = batch['captions_per_image']

            # Forward pass
            image_embeddings = model_vision(images)
            text_embeddings = model_text(captions)

            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            # Compute similarity matrix
            similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())

            # Compute loss using diagonal concept
            loss = diagonal_cross_entropy_loss(similarity_matrix, captions_per_image)

            # Compute accuracy
            accuracy = compute_accuracy(similarity_matrix, captions_per_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
        
        epoch_loss /= len(dataloader)
        epoch_accuracy /= len(dataloader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    return model_vision, model_text, train_losses, train_accuracies

def evaluate(model_vision, model_text, dataloader, device):
    model_vision.eval()
    model_text.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images, captions = batch['images'].to(device), batch['captions'].to(device)
            captions_per_image = batch['captions_per_image']
            
            # Forward pass
            image_embeddings = model_vision(images)
            text_embeddings = model_text(captions)
            
            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
            
            # Compute loss
            loss = diagonal_cross_entropy_loss(similarity_matrix, captions_per_image)
            
            # Compute accuracy
            accuracy = compute_accuracy(similarity_matrix, captions_per_image)
            
            total_loss += loss.item()
            total_accuracy += accuracy
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MSCOCODataset(split='train', transform=transform, tokenizer=tokenizer)
    val_dataset = MSCOCODataset(split='validation', transform=transform, tokenizer=tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

    vision_transformer = VisionTransformer().to(device)
    text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=64).to(device)
    
    optimizer = torch.optim.Adam(list(vision_transformer.parameters()) + list(text_transformer.parameters()))
    
    num_epochs = 10
    vision_transformer, text_transformer, train_losses, train_accuracies = train(num_epochs, train_dataloader, vision_transformer, text_transformer, optimizer, device)
    
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        val_loss, val_accuracy = evaluate(vision_transformer, text_transformer, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
    
    torch.save(vision_transformer.state_dict(), 'vision_transformer.pth')
    torch.save(text_transformer.state_dict(), 'text_transformer.pth')

if __name__ == '__main__':
    main()