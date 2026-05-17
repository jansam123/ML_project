import torch
import torch.nn as nn


class Model(nn.Module):
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

        self.score_net = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask, jet_features):
        # x: (B, N, F)
        # mask: (B, N)
        # jet_features: (B, J)

        h = self.particle_encoder(x)  # (B, N, H)

        mask_ = mask.unsqueeze(-1)  # (B, N, 1)
        h = h * mask_

        scores = self.score_net(h).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=1)  # (B, N)
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)  # (B, H)

        jet_h = self.jet_encoder(jet_features)  # (B, H)

        out = torch.cat([pooled, jet_h], dim=-1)
        logits = self.classifier(out)

        return logits
