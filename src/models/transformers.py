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
        
        self.pre_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.initparameters()

    def initparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        print(f"VisionTransformer input shape: {x.shape}")
        x = self.patch_embedding(x)
        print(f"After patch embedding shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        print(f"After flatten and transpose shape: {x.shape}")

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"Shape after adding CLS token: {x.shape}")
    
        x = x + self.pos_embedding
        
        x = self.pre_norm(x)
        x = self.encoder(x)
        
        return x[:, 0]  # Return the [CLS] token representation

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=49152, max_seq_len=76, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.pre_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.initparameters()

    def initparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        print(f"TextTransformer input shape: {input_ids.shape}")
        print(f"Input sequence length: {input_ids.shape[1]}")
        print(f"Max sequence length: {self.max_seq_len}")
        if input_ids.shape[1] > self.max_seq_len:
            print("Warning: Input sequence length exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        seq_len = x.size(1)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(position_ids)

        x = x + pos_emb

        x = self.pre_norm(x)

        if attention_mask is not None:
            attention_mask = attention_mask == 0
            print(f"Attention mask shape: {attention_mask.shape}")
            

        x = self.encoder(x, src_key_padding_mask=attention_mask)
        
        cls_output = x[:, 0]
        print(f"TextTransformer CLS token output shape: {cls_output.shape}")
        print(f"TextTransformer CLS token output (first 5 values): {cls_output[0, :5]}")
        return cls_output  # Return the first token representation

class CLIPModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.image_projection = nn.Sequential(
            nn.Linear(vision_model.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initparameters()

    def initparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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