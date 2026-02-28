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

    Architecture:
      EfficientNet-B0 backbone (ImageNet pretrained) → 1280-dim features
      → Dropout(0.4)  [already inside EfficientNet's classifier]
      → Linear(1280 → 1)  [binary output, no sigmoid — use BCEWithLogitsLoss]
    """

    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        in_features = self.backbone.classifier[1].in_features
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
        return self.backbone(x)