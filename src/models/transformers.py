import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def check_tensor(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or inf values")
    if tensor.dtype in [torch.float32, torch.float64]:
        print(f"{name} stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    else:
        print(f"{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, unique values={tensor.unique().numel()}")

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim) )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) )
        self.pos_embedding = nn.Embedding(num_patches+1, hidden_dim)
        self.cls_token = nn.Embedding(1, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

        

    def forward(self, x):
        #check_tensor(x, "VisionTransformer input")
        x = self.patch_embedding(x)
        #check_tensor(x, "After patch embedding")
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("NaN or Inf detected after patch embedding. Input stats:")
        #     print(f"Input min: {self.patch_embedding.weight.min():.4f}, max: {self.patch_embedding.weight.max():.4f}")
        #     print(f"Bias min: {self.patch_embedding.bias.min():.4f}, max: {self.patch_embedding.bias.max():.4f}" if self.patch_embedding.bias is not None else "No bias")

        x = x.flatten(2).transpose(1, 2)
        # check_tensor(x, "After flatten and transpose")

        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token(torch.zeros(batch_size, 1, dtype=torch.long, device=x.device))
        x = torch.cat((cls_tokens, x), dim=1)
        # check_tensor(x, "After adding CLS token vision")

        positions = torch.arange(seq_len + 1, dtype=torch.long, device=x.device)  # +1 for CLS token
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        x = x + pos_emb
        # check_tensor(x, "After adding positional embedding vision")
        
        x = self.pre_norm(x)
        # check_tensor(x, "After pre-norm vision")
        x = self.encoder(x)
        # check_tensor(x, "After encoder vision")
        
        cls_output = x[:, 0]
        # check_tensor(cls_output, "CLS token output vision")
        return cls_output

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
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask=None):
        check_tensor(input_ids, "TextTransformer input")
        if input_ids.shape[1] > self.max_seq_len:
            print("Warning: Input sequence length exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        # check_tensor(x, "After token embedding")
        seq_len = x.size(1)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(position_ids)
        # check_tensor(pos_emb, "Positional embedding")

        x = x + pos_emb
        # check_tensor(x, "After adding positional embedding")

        x = self.pre_norm(x)
        # check_tensor(x, "After pre-norm")

        if attention_mask is not None:
            attention_mask = attention_mask == 0
            check_tensor(attention_mask.float(), "Attention mask")

        x = self.encoder(x, src_key_padding_mask=attention_mask)
        # check_tensor(x, "After encoder")
        
        cls_output = x[:, 0]
        # check_tensor(cls_output, "CLS token output")
        return cls_output

class CLIPModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.image_projection = nn.Sequential(
            nn.Linear(vision_model.hidden_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model.hidden_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        
       
        
        # Initialize logit scale with the specified value
        
        # self.eps = 1e-8
        
        self.apply(self._init_weights)
        
        self.logit_scale_init_value = 2.6592
        self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)

        

    def _init_weights(self, module):
        
        if isinstance(module, (nn.Parameter, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        else:
            #  print (type(module))
            #  print("ERROR")
            #  raise ValueError
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, images, captions):
        image_features = self.vision_model(images)
        text_features = self.text_model(captions)
        
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        
        # check_tensor(image_features, "Image features before normalization")
        # check_tensor(text_features, "Text features before normalization")
        
        # image_features = F.normalize(image_features, dim=1, eps=self.eps)
        # text_features = F.normalize(text_features, dim=1, eps=self.eps)
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # check_tensor(image_features, "Image features after normalization")
        # check_tensor(text_features, "Text features after normalization")

        logit_scale = self.logit_scale.exp()
        similarity_matrix = logit_scale * torch.matmul(image_features, text_features.transpose(0, 1))
        
        check_tensor(similarity_matrix, "Similarity matrix")
        return similarity_matrix


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class VisionTransformer(nn.Module):
#     def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
#         super().__init__()
#         self.patch_size = patch_size
#         self.hidden_dim = hidden_dim
        
#         self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
#         num_patches = (image_size // patch_size) ** 2
#         self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
#         self.pre_norm = nn.LayerNorm(hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
#         self.apply(self._init_weights)
    
#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.Embedding):
#             nn.init.xavier_uniform_(module.weight)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.weight, 1.0)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, x):
#         print(f"VisionTransformer input shape: {x.shape}")
#         x = self.patch_embedding(x)
#         print(f"After patch embedding shape: {x.shape}")
#         x = x.flatten(2).transpose(1, 2)
#         print(f"After flatten and transpose shape: {x.shape}")

#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         print(f"Shape after adding CLS token: {x.shape}")
    
#         x = x + self.pos_embedding
        
#         x = self.pre_norm(x)
#         x = self.encoder(x)
        
#         return x[:, 0]  # Return the [CLS] token representation

# class TextTransformer(nn.Module):
#     def __init__(self, vocab_size=49152, max_seq_len=76, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.max_seq_len = max_seq_len
        
#         self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
#         self.pre_norm = nn.LayerNorm(hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, mlp_dim, norm_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.Embedding):
#             nn.init.xavier_uniform_(module.weight)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.weight, 1.0)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, input_ids, attention_mask=None):
#         print(f"TextTransformer input shape: {input_ids.shape}")
#         print(f"Input sequence length: {input_ids.shape[1]}")
#         print(f"Max sequence length: {self.max_seq_len}")
#         if input_ids.shape[1] > self.max_seq_len:
#             print("Warning: Input sequence length exceeds max_seq_len")

#         x = self.token_embedding(input_ids)
#         seq_len = x.size(1)

#         position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
#         pos_emb = self.pos_embedding(position_ids)

#         x = x + pos_emb

#         x = self.pre_norm(x)

#         if attention_mask is not None:
#             attention_mask = attention_mask == 0
#             print(f"Attention mask shape: {attention_mask.shape}")

#         x = self.encoder(x, src_key_padding_mask=attention_mask)
        
#         cls_output = x[:, 0]
#         print(f"TextTransformer CLS token output shape: {cls_output.shape}")
#         print(f"TextTransformer CLS token output (first 5 values): {cls_output[0, :5]}")
#         return cls_output  # Return the first token representation

# class CLIPModel(nn.Module):
#     def __init__(self, vision_model, text_model, projection_dim=512):
#         super().__init__()
#         self.vision_model = vision_model
#         self.text_model = text_model
#         self.image_projection = nn.Sequential(
#             nn.Linear(vision_model.hidden_dim, projection_dim),
#             nn.GELU(),
#             nn.LayerNorm(projection_dim)
#         )
#         self.text_projection = nn.Sequential(
#             nn.Linear(text_model.hidden_dim, projection_dim),
#             nn.GELU(),
#             nn.LayerNorm(projection_dim)
#         )
        
#         # Initialize logit scale with the specified value
#         self.logit_scale_init_value = 2.6592
#         self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.Embedding):
#             nn.init.xavier_uniform_(module.weight)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.weight, 1.0)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, images, captions):
#         image_features = self.vision_model(images)
#         text_features = self.text_model(captions)
        
#         image_features = self.image_projection(image_features)
#         text_features = self.text_projection(text_features)
        
#         image_features = F.normalize(image_features, dim=1)
#         text_features = F.normalize(text_features, dim=1)

#         logit_scale = self.logit_scale.exp()
#         similarity_matrix = logit_scale * torch.matmul(image_features, text_features.transpose(0, 1))

#         return similarity_matrix