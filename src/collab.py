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
    images = []
    captions = []
    for item in batch:
        if item['image'] is not None:
            images.append(item['image'])
            # Randomly select one caption for this image
            selected_caption = random.choice(item['captions'])
            captions.append(selected_caption)
    
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return {'images': images, 'captions': captions}

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def compute_accuracy(logits: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == torch.arange(len(logits), device=logits.device)).float().mean().item()

def train_epoch(dataloader, clip_model, optimizer, device, epoch, warmup_steps):
    clip_model.train()
    epoch_loss = 0.0
    epoch_i2t_accuracy = 0.0
    epoch_t2i_accuracy = 0.0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        images, captions = batch['images'].to(device), batch['captions'].to(device)

        if len(optimizer.state) == 0:
            step = 0
        else:
            step = optimizer.state[optimizer.param_groups[0]['params'][0]].get('step', 0)
        
        step = int(step)
        
        if step < warmup_steps:
            lr_scale = min(1., float(step + 1) / warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-5

        similarity_matrix = clip_model(images, captions)

        i2t_loss = contrastive_loss(similarity_matrix)
        t2i_loss = contrastive_loss(similarity_matrix.t())
        loss = (i2t_loss + t2i_loss) / 2

        i2t_accuracy = compute_accuracy(similarity_matrix)
        t2i_accuracy = compute_accuracy(similarity_matrix.t())

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_i2t_accuracy += i2t_accuracy
        epoch_t2i_accuracy += t2i_accuracy

        wandb.log({
            "epoch": epoch + 1,
            "train_batch_loss": loss.item(),
            "train_batch_i2t_accuracy": i2t_accuracy,
            "train_batch_t2i_accuracy": t2i_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }, step=step)

    num_batches = len(dataloader)
    epoch_loss /= num_batches
    epoch_i2t_accuracy /= num_batches
    epoch_t2i_accuracy /= num_batches
    epoch_accuracy = (epoch_i2t_accuracy + epoch_t2i_accuracy) / 2

    return epoch_loss, epoch_accuracy, epoch_i2t_accuracy, epoch_t2i_accuracy


def evaluate(clip_model, dataloader, device):
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
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_i2t_accuracy = total_i2t_accuracy / num_batches
    avg_t2i_accuracy = total_t2i_accuracy / num_batches
    avg_accuracy = (avg_i2t_accuracy + avg_t2i_accuracy) / 2

    return avg_loss, avg_accuracy, avg_i2t_accuracy, avg_t2i_accuracy

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    # train_dataset = full_train_dataset
    val_dataset = Subset(full_val_dataset, range(num_test_samples))

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # vision_transformer = VisionTransformer().to(device)
    # text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=64).to(device)
    # clip_model = CLIPModel(vision_transformer, text_transformer).to(device)

    vision_transformer = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072).to(device)
    text_transformer = TextTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=76, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072).to(device)
    clip_model = CLIPModel(vision_transformer, text_transformer, projection_dim=512).to(device)
    
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    
    num_epochs = 100
    warmup_steps = 10000
    print(f"Starting training for {num_epochs} epochs")
    experiment_name = 'CLS_token_Training'
    wandb.init(project="Contrastive-loss",
               name = experiment_name,
         config={
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_train_samples": len(train_dataset),
        "num_test_samples": num_test_samples,
        "warmup_steps": warmup_steps
    })

    best_val_accuracy = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        train_loss, train_accuracy, train_i2t_accuracy, train_t2i_accuracy = train_epoch(
            train_dataloader, clip_model, optimizer, device, epoch, warmup_steps)
        
        # Validation
        val_loss, val_accuracy, val_i2t_accuracy, val_t2i_accuracy = evaluate(
            clip_model, val_dataloader, device)
        
        epoch_time = time.time() - epoch_start_time
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_epoch_loss": train_loss,
            "train_epoch_accuracy": train_accuracy,
            "train_epoch_i2t_accuracy": train_i2t_accuracy,
            "train_epoch_t2i_accuracy": train_t2i_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_i2t_accuracy": val_i2t_accuracy,
            "val_t2i_accuracy": val_t2i_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "epoch_time": epoch_time,
        })

        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        # print(f"Train Loss: {train_loss:.4f}, Train I2T Acc: {train_i2t_accuracy:.4f}, Train T2I Acc: {train_t2i_accuracy:.4f}")
        # print(f"Val Loss: {val_loss:.4f}, Val I2T Acc: {val_i2t_accuracy:.4f}, Val T2I Acc: {val_t2i_accuracy:.4f}")
        # print(f"Epoch Time: {epoch_time:.2f} seconds")

    torch.save(clip_model.state_dict(), 'clip_model.pth')
    wandb.save('clip_model.pth')

    wandb.finish()

if __name__ == '__main__':
    main()