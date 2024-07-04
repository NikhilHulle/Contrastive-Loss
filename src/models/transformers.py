import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim),
            num_layers
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask=None):
        print("Initial input_ids shape:", input_ids.shape)
        print("Initial attention_mask shape:", attention_mask.shape if attention_mask is not None else None)

        x = self.token_embedding(input_ids)
        seq_len = x.size(1)

        if seq_len <= self.max_seq_len:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(self.pos_embedding.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)

        x = x + pos_emb

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            print("Boolean attention_mask shape:", attention_mask.shape)

        x = self.encoder_layers(x, src_key_padding_mask=attention_mask)
        print("After encoder layers shape:", x.shape)

        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        print("Final output shape:", x.shape)

        return x

class CLIPModel(nn.Module):
    def __init__(self, vision_model, text_model):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, captions):
        image_features = self.vision_model(images)
        text_features = self.text_model(captions)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        similarity_matrix = logit_scale * image_features @ text_features.t()

        return similarity_matrix