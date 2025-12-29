import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from src.face_transformer import FaceTransformModule

per_img_token_list = ['!','@','#','$','%','^','&','(',')']

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    #assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

class EmbeddingManager(nn.Module):
    def __init__(
            self,
            token_model,
            dtype,
            placeholder_strings=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            attention=None,
            dim=768,
            weight=1.0,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = {}
        self.string_to_token_dict_two = {}
        self.init = True
        self.cond_stage_model = token_model
        self.progressive_words = progressive_words
        self.progressive_counter = 0
        self.max_vectors_per_token = num_vectors_per_token
        self.dtype = dtype
        self.dim = dim

        get_token_for_string = partial(get_clip_token_for_string, self.cond_stage_model)
        if attention is None:
            self.attention = FaceTransformModule(768, 8, 64, dropout = 0.05, context_dim=512)
        else:
            self.attention = attention
        self.linear_model = nn.Linear(768, dim)
        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)
    
        for _, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)
            self.string_to_token_dict[placeholder_string] = token
        self.face_weight = weight

    def forward(
            self,
            tokenized_text,
            embedded_text,            
            img_embedding,
            id_embedding,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        for idx, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding, _, _ = self.attention(img_embedding.view(b,1,768).to(device), id_embedding.view(b,1,512).to(device))
            placeholder_embedding = self.linear_model(placeholder_embedding)
            # controlling identity weights through the face_weight hyperparameter,
            placeholder_embedding = placeholder_embedding * self.face_weight
            # replace the placeholder's text embedding
            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            embedded_text[placeholder_idx] = placeholder_embedding.to(self.dtype)
        return embedded_text

    def embedding_parameters(self):
        return list(self.attention.parameters()) + list(self.linear_model.parameters())
    
    def save(self, ckpt_path):
        torch.save({
                    "string_to_token": self.string_to_token_dict,
                    "attention": self.attention,
                    'linear': self.linear_model,
                    }, ckpt_path)
    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print('find keys:',ckpt.keys())

        self.string_to_token_dict = ckpt["string_to_token"]

        if 'attention' in ckpt.keys():
            self.attention = ckpt["attention"]
        if 'linear' in ckpt.keys():
            self.linear_model = ckpt["linear"]