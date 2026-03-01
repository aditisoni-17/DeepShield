import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeCNN(nn.Module):
    """
    Deepfake detector built on a pretrained EfficientNet-B0 backbone.

    Training strategy (two phases managed by train.py):
      Phase 1 — backbone frozen, only the classifier head trains.
                 Quickly learns to map EfficientNet features → fake/real.
      Phase 2 — last two MBConv blocks + classifier unfrozen with a small LR.
                 Backbone subtly adapts to deepfake-specific texture cues.

    Architecture (use_frequency=False, default):
      EfficientNet-B0 backbone (ImageNet pretrained) → 1280-dim features
      → Dropout(0.4) → Linear(1280 → 1)

    Architecture (use_frequency=True):
      EfficientNet-B0 → 1280-dim
      FrequencyBranch (FFT spectrum) → 128-dim
      Fused → Linear(1408 → 256) → ReLU → Dropout(0.4) → Linear(256 → 1)

    Pass use_frequency=True only when training from scratch with that flag,
    since it changes the model's parameter shape (incompatible with a
    checkpoint trained without it).
    """

    def __init__(self, use_frequency: bool = False):
        super(DeepfakeCNN, self).__init__()
        self.use_frequency = use_frequency

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features  # 1280

        if use_frequency:
            from model.frequency_branch import FrequencyBranch
            self.freq_branch = FrequencyBranch()
            # Remove EfficientNet's own classifier; we fuse manually
            self.backbone.classifier = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(in_features + 128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256, 1),
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 1),
            )

        # Start with backbone fully frozen; train.py calls unfreeze_top_blocks()
        # after the warm-up phase
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone (feature extractor) parameters."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self):
        """
        Unfreeze the last two MBConv blocks for fine-tuning.
        Called by train.py after the classifier head has warmed up.
        blocks[7] and blocks[8] are the deepest feature layers in EfficientNet-B0.
        """
        for block in [self.backbone.features[7], self.backbone.features[8]]:
            for param in block.parameters():
                param.requires_grad = True
        print("Unfroze EfficientNet blocks 7 & 8 for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_frequency:
            spatial_features = self.backbone(x)        # (B, 1280)
            freq_features = self.freq_branch(x)        # (B, 128)
            fused = torch.cat([spatial_features, freq_features], dim=1)  # (B, 1408)
            return self.head(fused)
        return self.backbone(x)