import torch
import torch.nn as nn


def masked_mean_pool(x, mask):
    mask = mask.to(dtype=x.dtype).unsqueeze(-1)
    x = x * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return x.sum(dim=1) / denom


def masked_max_pool(x, mask):
    mask = mask.bool().unsqueeze(-1)
    neg = torch.finfo(x.dtype).min
    x = torch.where(mask, x, torch.full_like(x, neg))
    return torch.max(x, dim=1).values


def pool(x, mask, pooling_type="mean"):
    if pooling_type == "max":
        return masked_max_pool(x, mask)
    elif pooling_type == "mean":
        return masked_mean_pool(x, mask)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


class JetOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        jet_feature_dim = config.get("jet_feature_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        jet_h = self.jet_encoder(jet_features)
        return self.classifier(jet_h)


class ParticleOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        self.pooling_type = config.get("pooling_type", "mean")

        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features=None):
        h = self.particle_encoder(x)
        pooled = pool(h, mask, self.pooling_type)
        return self.classifier(pooled)


class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        jet_feature_dim = config.get("jet_feature_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        self.pooling_type = config.get("pooling_type", "mean")

        self.particle_encoder = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        h = self.particle_encoder(x)
        pooled = pool(h, mask, self.pooling_type)

        jet_h = self.jet_encoder(jet_features)

        return self.classifier(torch.cat([pooled, jet_h], dim=-1))

import torch
import torch.nn as nn


def masked_mean_pool(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)
    x = x * mask
    summed = x.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def masked_max_pool(x, mask):
    mask = mask.bool().unsqueeze(-1)
    x = x.masked_fill(~mask, torch.finfo(x.dtype).min)
    return x.max(dim=1).values


def pool(x, mask, pooling_type="mean"):
    if pooling_type == "max":
        return masked_max_pool(x, mask)
    return masked_mean_pool(x, mask)


class ParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        jet_feature_dim = config.get("jet_feature_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 2)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)

        self.pooling_type = config.get("pooling_type", "mean")
        self.use_jet_features = config.get("use_jet_features", True)

        self.input_embed = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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

        if self.use_jet_features:
            self.jet_encoder = nn.Sequential(
                nn.Linear(jet_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_in = hidden_dim * 2
        else:
            self.jet_encoder = None
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features=None):
        valid_mask = mask.bool()

        # Prevent fully-masked rows from causing attention issues.
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = True

        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        h = self.input_embed(x)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)

        pooled = pool(h, valid_mask, self.pooling_type)

        if self.use_jet_features:
            if jet_features is None:
                raise ValueError("jet_features is required when use_jet_features=True")
            jet_h = self.jet_encoder(jet_features)
            out = torch.cat([pooled, jet_h], dim=-1)
        else:
            out = pooled

        return self.classifier(out)