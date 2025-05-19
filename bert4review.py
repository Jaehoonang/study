import torch
import torch.nn as nn

class transformerclassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.randn(1, max_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.position[:, :x.size(1), :]
        x = x.permute(1, 0 ,2)
        x = self.transformer(x)
        x = x[0]
        return self.classifier(x)