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
        
        self.pre_norm = nn.LayerNorm(hidden_dim) # change 1
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True) # change 2
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        x = self.pre_norm(x)
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x[:, 0]  # Return the [CLS] token representation

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=49152, max_seq_len=76, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072): # change 3
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        self.pre_norm = nn.LayerNorm(hidden_dim) # change 4
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True), # change 5
            num_layers
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        seq_len = x.size(1)

        if seq_len <= self.max_seq_len:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(self.pos_embedding.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)

        x = x + pos_emb

        x = self.pre_norm(x) # change 6

        if attention_mask is not None: # change 7
            attention_mask = attention_mask.bool()
            # Create a causal mask for masked self-attention**
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(input_ids.device)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) & ~causal_mask.unsqueeze(0)

        x = self.encoder_layers(x, src_mask=attention_mask)

        return x[:, 0]  # Return the [CLS] token representation

class CLIPModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):   # change 8
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        # self.image_projection = nn.Linear(vision_model.hidden_dim, projection_dim) # change 9
        # self.text_projection = nn.Linear(text_model.hidden_dim, projection_dim) # change 10
        self.image_projection = nn.Sequential(
            nn.Linear(vision_model.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, captions):
        image_features = self.vision_model(images)
        text_features = self.text_model(captions)
        
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        similarity_matrix = logit_scale * image_features @ text_features.t()

        return similarity_matrix