import torch
import torch.nn as nn


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


import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
        Inputs:
        x:
            Particle-level features.
            Shape: (B, N, Fp)

        mask:
            Particle mask.
            Shape: (B, N)
            1 = real particle
            0 = padding

        jet_features:
            Optional global jet features.
            Shape: (B, Fj)

    Output:
        logits:
            Class scores.
            Shape: (B, num_classes)

    """

    def __init__(self, config: dict):
        super().__init__()

        self.num_particle_features = config.get("num_particle_features", 9)
        self.num_jet_features = config.get("num_jet_features", 9)
        self.num_classes = config.get("num_classes", 4)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.dropout = config.get("dropout", 0.2)

        self.use_jet_features = config.get("use_jet_features", False)
        self.use_transformer = config.get("use_transformer", True)

        self.num_heads = config.get("num_heads", 4)
        self.num_transformer_layers = config.get("num_transformer_layers", 2)
        self.transformer_dim = config.get("transformer_dim", 512)

        # 1. Embed each particle independently.
        #
        # Input:
        #     x has shape (B, N, num_particle_features)
        #
        # Output:
        #     particle_embeddings has shape (B, N, hidden_dim)
        self.particle_embedding = nn.Sequential(
            nn.Linear(self.num_particle_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # 2. Transformer block.
        #
        # particles interact with each other.
        # The padding mask tells the Transformer which particles are fake padding.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.transformer_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_transformer_layers,
        )

        # This is useful if later we want to compare:
        #     event_embedding from particles
        # with:
        #     jet_embedding from global jet features
        self.jet_projection = nn.Sequential(
            nn.Linear(self.num_jet_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # If use_jet_features=False:
        #     classifier input = event_embedding
        #
        # If use_jet_features=True:
        #     classifier input = event_embedding + jet_embedding
        fusion_dim = self.hidden_dim

        if self.use_jet_features:
            fusion_dim += self.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x, mask, jet_features=None, return_embeddings=False):
        """
        x:
            Shape: (B, N, Fp)

        mask:
            Shape: (B, N)
            1 = real particle
            0 = padding

        jet_features:
            Shape: (B, Fj), optional

        return_embeddings:
            If False, returns only logits.
            If True, returns logits plus embeddings useful for comparisons.
        """

        # Make sure mask is on the same device as x.
        mask = mask.to(x.device)

        if jet_features is not None:
            jet_features = jet_features.to(x.device)

        # Embed each particle.
        particle_embeddings = self.particle_embedding(x)
        # Shape: (B, N, hidden_dim)

        # Transformer wants True for the positions that must be ignored.
        padding_mask = mask == 0
        # Shape: (B, N)

        if self.use_transformer:
            particle_embeddings = self.transformer(
                particle_embeddings,
                src_key_padding_mask=padding_mask,
            )

        # Remove padded particles before pooling.
        mask_expanded = mask.unsqueeze(-1).to(particle_embeddings.dtype)
        # Shape: (B, N, 1)

        particle_embeddings = particle_embeddings * mask_expanded

        # Pool all real particles into one event-level vector.
        summed = particle_embeddings.sum(dim=1)
        # Shape: (B, hidden_dim)

        n_particles = mask_expanded.sum(dim=1).clamp(min=1.0)
        # Shape: (B, 1)

        event_embedding = summed / n_particles
        # Shape: (B, hidden_dim)

        # This is the input to the classifier.
        classifier_input = event_embedding

        # Build jet embedding if jet_features are available.
        jet_embedding = None

        if jet_features is not None:
            jet_embedding = self.jet_projection(
                jet_features.to(event_embedding.dtype)
            )
            # Shape: (B, hidden_dim)

        # Optionally use jet features for classification.
        if self.use_jet_features:
            if jet_embedding is None:
                raise ValueError(
                    "use_jet_features=True, but jet_features was not provided."
                )

            classifier_input = torch.cat(
                [event_embedding, jet_embedding],
                dim=1,
            )
            # Shape: (B, 2 * hidden_dim)

        # Final class prediction.
        logits = self.classifier(classifier_input)
        # Shape: (B, num_classes)

        # The Trainer expects logits directly for CrossEntropyLoss.
        if not return_embeddings:
            return logits

        #  output for future analysis/comparison.
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