import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import DistilBertTokenizerFast
from custom_datasets.mscoco_dataset import MSCOCODataset
from models.transformers import VisionTransformer, TextTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time



def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch if item['image'] is not None])
    all_captions = [item['captions'] for item in batch]
    
    captions_per_image = [len(captions) for captions in all_captions]
    
    flattened_captions = [caption for sublist in all_captions for caption in sublist]
    padded_captions = pad_sequence(flattened_captions, batch_first=True, padding_value=0)
    
    return {'images': images, 'captions': padded_captions, 'captions_per_image': captions_per_image}

def diagonal_cross_entropy_loss(similarity_matrix, captions_per_image, temperature=0.216):
    device = similarity_matrix.device
    batch_size = similarity_matrix.size(0)
    
    # Ensure captions_per_image is a tensor
    if not isinstance(captions_per_image, torch.Tensor):
        captions_per_image = torch.tensor(captions_per_image, device=device)
    
    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=device).repeat_interleave(captions_per_image)
    
    # Reshape similarity matrix to match labels
    similarity_matrix = similarity_matrix.repeat_interleave(captions_per_image, dim=0) / temperature
    
    loss_i2t = F.cross_entropy(similarity_matrix, labels)
    loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
    
    return (loss_i2t + loss_t2i) / 2

def compute_accuracy(similarity_matrix, captions_per_image):
    device = similarity_matrix.device
    batch_size = similarity_matrix.size(0)
    
    # Ensure captions_per_image is a tensor
    if not isinstance(captions_per_image, torch.Tensor):
        captions_per_image = torch.tensor(captions_per_image, device=device)
    
    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=device).repeat_interleave(captions_per_image)
    
    # Reshape similarity matrix to match labels
    similarity_matrix = similarity_matrix.repeat_interleave(captions_per_image, dim=0)
    
    # Compute predictions
    _, predicted = similarity_matrix.max(1)
    
    # Compute accuracy
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    
    return correct / total

def train(num_epochs, dataloader, model_vision, model_text, optimizer, device, warmup_steps=1000):
    model_vision.train()
    model_text.train()

    train_losses = []
    train_accuracies = []

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, captions = batch['images'].to(device), batch['captions'].to(device)
            captions_per_image = torch.tensor(batch['captions_per_image'], device=device)

            if global_step < warmup_steps:
                lr_scale = min(1., float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * 1e-5

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

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            global_step += 1
        
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, captions = batch['images'].to(device), batch['captions'].to(device)
            captions_per_image = torch.tensor(batch['captions_per_image'], device=device)
            
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
    
    # Save the figure
    plt.savefig('training_metrics.png')
    print(f"Figure saved as 'training_metrics.png'")
    
    # Try to display the figure
    try:
        plt.show()
    except Exception as e:
        print(f"Couldn't display the figure. Error: {e}")
    
    # Close the figure to free up memory
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load a subset of the dataset for testing
    num_test_samples = 100
    num_train_samples = 300
    full_train_dataset = MSCOCODataset(split='train', transform=transform, tokenizer=tokenizer)
    full_val_dataset = MSCOCODataset(split='validation', transform=transform, tokenizer=tokenizer)

    train_dataset = Subset(full_train_dataset, range(num_train_samples))
    val_dataset = Subset(full_val_dataset, range(num_test_samples))

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    vision_transformer = VisionTransformer().to(device)
    text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=64).to(device)
    
    optimizer = torch.optim.Adam(list(vision_transformer.parameters()) + list(text_transformer.parameters()), lr=1e-5)
    
    num_epochs = 500
    print(f"Starting training for {num_epochs} epochs with {num_test_samples} samples")
    
    start_time = time.time()
    vision_transformer, text_transformer, train_losses, train_accuracies = train(num_epochs, train_dataloader, vision_transformer, text_transformer, optimizer, device)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    val_losses = []
    val_accuracies = []
    print("Starting validation")
    start_time = time.time()
    for epoch in range(num_epochs):
        val_loss, val_accuracy = evaluate(vision_transformer, text_transformer, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    val_time = time.time() - start_time
    print(f"Validation completed in {val_time:.2f} seconds")

    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # print("Displaying the saved image:")
    # display(Image('training_metrics.png'))
    
    torch.save(vision_transformer.state_dict(), 'vision_transformer.pth')
    torch.save(text_transformer.state_dict(), 'text_transformer.pth')

if __name__ == '__main__':
    main()