import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import DistilBertTokenizerFast
from custom_datasets.mscoco_dataset import MSCOCODataset
from models.transformers import VisionTransformer, TextTransformer, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb
import random

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch if item['image'] is not None])
    all_captions = [item['captions'] for item in batch]
    
    # Randomly select one caption per image
    selected_captions = [random.choice(captions) for captions in all_captions]
    
    padded_captions = pad_sequence(selected_captions, batch_first=True, padding_value=0)
    
    return {'images': images, 'captions': padded_captions}

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def compute_accuracy(logits: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == torch.arange(len(logits), device=logits.device)).float().mean().item()

def train(num_epochs, dataloader, clip_model, optimizer, device, warmup_steps=1000):
    clip_model.train()

    train_losses = []
    train_accuracies = []

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_i2t_accuracy = 0.0
        epoch_t2i_accuracy = 0.0
        epoch_start_time = time.time()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, captions = batch['images'].to(device), batch['captions'].to(device)

            if global_step < warmup_steps:
                lr_scale = min(1., float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * 1e-5

            similarity_matrix = clip_model(images, captions)

            # Image-to-text contrastive loss
            i2t_loss = contrastive_loss(similarity_matrix)
            # Text-to-image contrastive loss
            t2i_loss = contrastive_loss(similarity_matrix.t())
            
            # Total loss is the average of both directions
            loss = (i2t_loss + t2i_loss) / 2

            # Compute accuracy for both directions
            i2t_accuracy = compute_accuracy(similarity_matrix)
            t2i_accuracy = compute_accuracy(similarity_matrix.t())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_i2t_accuracy += i2t_accuracy
            epoch_t2i_accuracy += t2i_accuracy
            global_step += 1

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": loss.item(),
                "train_i2t_accuracy": i2t_accuracy,
                "train_t2i_accuracy": t2i_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }, step=global_step)
        
        epoch_loss /= len(dataloader)
        epoch_i2t_accuracy /= len(dataloader)
        epoch_t2i_accuracy /= len(dataloader)
        epoch_time = time.time() - epoch_start_time
        train_losses.append(epoch_loss)
        train_accuracies.append((epoch_i2t_accuracy + epoch_t2i_accuracy) / 2)
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, I2T Accuracy: {epoch_i2t_accuracy:.4f}, T2I Accuracy: {epoch_t2i_accuracy:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": epoch_loss,
            "epoch_train_i2t_accuracy": epoch_i2t_accuracy,
            "epoch_train_t2i_accuracy": epoch_t2i_accuracy,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    return clip_model, train_losses, train_accuracies

def evaluate(clip_model, dataloader, device, epoch):
    clip_model.eval()
    
    total_loss = 0.0
    total_i2t_accuracy = 0.0
    total_t2i_accuracy = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, captions = batch['images'].to(device), batch['captions'].to(device)
            
            similarity_matrix = clip_model(images, captions)
            
            i2t_loss = contrastive_loss(similarity_matrix)
            t2i_loss = contrastive_loss(similarity_matrix.t())
            loss = (i2t_loss + t2i_loss) / 2
            
            i2t_accuracy = compute_accuracy(similarity_matrix)
            t2i_accuracy = compute_accuracy(similarity_matrix.t())
        
            total_loss += loss.item()
            total_i2t_accuracy += i2t_accuracy
            total_t2i_accuracy += t2i_accuracy
    
    avg_loss = total_loss / len(dataloader)
    avg_i2t_accuracy = total_i2t_accuracy / len(dataloader)
    avg_t2i_accuracy = total_t2i_accuracy / len(dataloader)
    avg_accuracy = (avg_i2t_accuracy + avg_t2i_accuracy) / 2

    wandb.log({
        "epoch": epoch + 1,
        "val_loss": avg_loss,
        "val_i2t_accuracy": avg_i2t_accuracy,
        "val_t2i_accuracy": avg_t2i_accuracy,
        "val_accuracy": avg_accuracy,
    })
    
    return avg_loss, avg_accuracy, avg_i2t_accuracy, avg_t2i_accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_test_samples = 128
    num_train_samples = 640
    full_train_dataset = MSCOCODataset(split='train', transform=transform, tokenizer=tokenizer)
    full_val_dataset = MSCOCODataset(split='validation', transform=transform, tokenizer=tokenizer)

    train_dataset = Subset(full_train_dataset, range(num_train_samples))
    val_dataset = Subset(full_val_dataset, range(num_test_samples))

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    vision_transformer = VisionTransformer().to(device)
    text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=64).to(device)
    clip_model = CLIPModel(vision_transformer, text_transformer).to(device)
    
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    
    num_epochs = 500
    warmup_steps = 1000
    print(f"Starting training for {num_epochs} epochs")
    wandb.init(project="Contrastive-loss", config={
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "warmup_steps": warmup_steps
    })
    start_time = time.time()
    clip_model, train_losses, train_accuracies = train(num_epochs, train_dataloader, clip_model, optimizer, device, warmup_steps)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    val_losses = []
    val_accuracies = []
    print("Starting validation")
    best_val_accuracy = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        val_loss, val_accuracy, val_i2t_accuracy, val_t2i_accuracy = evaluate(clip_model, val_dataloader, device, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        wandb.log({
            "best_val_accuracy": best_val_accuracy,
        })
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation I2T Accuracy: {val_i2t_accuracy:.4f}, Validation T2I Accuracy: {val_t2i_accuracy:.4f}")
    val_time = time.time() - start_time
    print(f"Validation completed in {val_time:.2f} seconds")
    
    torch.save(clip_model.state_dict(), 'clip_model.pth')
    wandb.save('clip_model.pth')

    wandb.finish()

if __name__ == '__main__':
    main()