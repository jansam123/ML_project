import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)  # (B, N, 1)
    x = x * mask
    summed = x.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def masked_max_pool(x, mask):
    mask = mask.unsqueeze(-1).bool()
    x = x.masked_fill(~mask, float("-inf"))
    return x.max(dim=1).values


def pool(x, mask, pooling_type="mean"):
    if pooling_type == "max":
        return masked_max_pool(x, mask)
    return masked_mean_pool(x, mask)


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

        out = torch.cat([pooled, jet_h], dim=-1)
        return self.classifier(out)


class ParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        particle_dim = config.get("particle_dim", 9)
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        num_classes = config.get("num_classes", 4)
        dropout = config.get("dropout", 0.1)
        self.pooling_type = config.get("pooling_type", "max")

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
        h = self.input_embed(x)

        padding_mask = (mask == 0)
        h = self.encoder(h, src_key_padding_mask=padding_mask)

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







import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(x, mask):
    """
    x:    (B, N, H)
    mask: (B, N), 1 = real particle, 0 = padding
    """
    mask = mask.unsqueeze(-1).to(x.dtype)
    x = x * mask
    summed = x.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def masked_max_pool(x, mask):
    """
    x:    (B, N, H)
    mask: (B, N), 1 = real particle, 0 = padding
    """
    mask_bool = mask.unsqueeze(-1).bool()
    x = x.masked_fill(~mask_bool, float("-inf"))
    pooled = x.max(dim=1).values

    # Safety in case a jet has only padding.
    pooled = torch.nan_to_num(pooled, neginf=0.0)
    return pooled


def pool_particles(x, mask, pooling_type="mean"):
    if pooling_type == "max":
        return masked_max_pool(x, mask)
    elif pooling_type == "mean":
        return masked_mean_pool(x, mask)
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")


class Model(nn.Module):
    """
    Improved particle-level Transformer model.

    Inputs:
        x:
            Particle-level features.
            Shape: (B, N, num_particle_features)

        mask:
            Particle mask.
            Shape: (B, N)
            1 = real particle
            0 = padding

        jet_features:
            Optional global jet features.
            Shape: (B, num_jet_features)

    Output:
        logits:
            Class scores.
            Shape: (B, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.num_particle_features = config.get("num_particle_features", 9)
        self.num_jet_features = config.get("num_jet_features", 9)
        self.num_classes = config.get("num_classes", 10)

        self.hidden_dim = config.get("hidden_dim", 256)
        self.dropout = config.get("dropout", 0.1)

        self.use_jet_features = config.get("use_jet_features", False)
        self.use_transformer = config.get("use_transformer", True)

        self.num_heads = config.get("num_heads", 8)
        self.num_transformer_layers = config.get("num_transformer_layers", 2)

        # Important:
        # If hidden_dim=256, a good default is 4 * hidden_dim = 1024.
        self.transformer_dim = config.get("transformer_dim", self.hidden_dim * 4)

        # mean or max
        self.pooling_type = config.get("pooling_type", "mean")

        # Optional final normalization after the Transformer.
        self.final_norm = nn.LayerNorm(self.hidden_dim)

        # 1. Particle embedding.
        # Stronger than a single Linear layer, but still simple.
        self.particle_embedding = nn.Sequential(
            nn.Linear(self.num_particle_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # 2. Transformer encoder.
        # norm_first=True is important: it usually makes Transformer training more stable.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.transformer_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_transformer_layers,
        )

        # 3. Jet feature projection.
        # Used only if use_jet_features=True.
        self.jet_projection = nn.Sequential(
            nn.Linear(self.num_jet_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # 4. Final classifier.
        # Deeper than the previous one, more similar to ParticleTransformer.
        fusion_dim = self.hidden_dim
        if self.use_jet_features:
            fusion_dim += self.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x, mask, jet_features=None, return_embeddings=False):
        """
        x:
            Shape: (B, N, num_particle_features)

        mask:
            Shape: (B, N)
            1 = real particle
            0 = padding

        jet_features:
            Shape: (B, num_jet_features), optional

        return_embeddings:
            If False, returns only logits.
            If True, returns logits plus useful embeddings.
        """

        mask = mask.to(x.device)

        if jet_features is not None:
            jet_features = jet_features.to(x.device)

        # Embed particles.
        particle_embeddings = self.particle_embedding(x)

        # Transformer padding mask:
        # True = ignore this particle.
        padding_mask = mask == 0

        if self.use_transformer:
            particle_embeddings = self.transformer(
                particle_embeddings,
                src_key_padding_mask=padding_mask,
            )

        # Final normalization after Transformer.
        particle_embeddings = self.final_norm(particle_embeddings)

        # Pool particles into one event-level embedding.
        event_embedding = pool_particles(
            particle_embeddings,
            mask,
            pooling_type=self.pooling_type,
        )

        classifier_input = event_embedding

        # Optional jet feature branch.
        jet_embedding = None

        if self.use_jet_features:
            if jet_features is None:
                raise ValueError(
                    "use_jet_features=True, but jet_features was not provided."
                )

            jet_embedding = self.jet_projection(
                jet_features.to(event_embedding.dtype)
            )

            classifier_input = torch.cat(
                [event_embedding, jet_embedding],
                dim=1,
            )

        elif jet_features is not None:
            # We compute it only for analysis if return_embeddings=True.
            if return_embeddings:
                jet_embedding = self.jet_projection(
                    jet_features.to(event_embedding.dtype)
                )

        logits = self.classifier(classifier_input)

        if not return_embeddings:
            return logits

        output = {
            "logits": logits,
            "event_embedding": event_embedding,
            "event_embedding_normalized": F.normalize(event_embedding, dim=1),
            "jet_embedding": jet_embedding,
        }

        if jet_embedding is not None:
            output["jet_embedding_normalized"] = F.normalize(
                jet_embedding,
                dim=1,
            )
        else:
            output["jet_embedding_normalized"] = None

        return output