import torch
import torch.nn as nn
# from particle_transformer import *


class JetOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        jet_feature_dim = config.get("jet_feature_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)

        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        jet_h = self.jet_encoder(jet_features)
        logits = self.classifier(jet_h)
        return logits



class ParticleOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)

        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        h = self.particle_encoder(x)  # (B, N, H)

        mask_ = mask.unsqueeze(-1).to(dtype=h.dtype)
        h = h * mask_

        pooled = h.sum(dim=1) / mask_.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(pooled)
        return logits

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        jet_feature_dim = config.get("jet_feature_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)

        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        h = self.particle_encoder(x)

        mask_ = mask.unsqueeze(-1).to(dtype=h.dtype)
        h = h * mask_

        pooled = h.sum(dim=1) / mask_.sum(dim=1).clamp(min=1.0)

        jet_h = self.jet_encoder(jet_features)

        out = torch.cat([pooled, jet_h], dim=-1)
        logits = self.classifier(out)

        return logits


class ParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        self.input_embed = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features=None):
        h = self.input_embed(x)

        # True = ignore (PyTorch convention)
        key_padding_mask = (mask == 0)

        h = self.encoder(h, key_padding_mask=key_padding_mask)

        mask_ = mask.unsqueeze(-1).to(h.dtype)
        h = h * mask_

        pooled = h.sum(dim=1) / mask_.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled)

class ParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        # --- input embedding ---
        self.input_embed = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        """
        x: (B, N, F)
        mask: (B, N)
        """

        B, N, _ = x.shape

        # 1. embed particles
        h = self.input_embed(x)  # (B, N, H)

        # 2. build attention mask (True = ignore)
        attn_mask = (mask == 0)  # (B, N)

        # PyTorch expects: (B, N) bool mask
        h = self.encoder(h, src_key_padding_mask=attn_mask)

        # 3. masked mean pooling
        mask_ = mask.unsqueeze(-1).to(h.dtype)
        h = h * mask_

        pooled = h.sum(dim=1) / mask_.sum(dim=1).clamp(min=1.0)

        # 4. classification
        logits = self.classifier(pooled)

        return logits


