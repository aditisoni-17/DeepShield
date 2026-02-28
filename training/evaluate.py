"""
Evaluate the trained model on the test set and report metrics.
Run after training: python -m training.evaluate
"""

import torch

from model.cnn_model import DeepfakeCNN
from training.dataset import get_dataloaders
from training.metrics import calculate_metrics


def evaluate(model_path="saved_models/best_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders()

    model = DeepfakeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs.sigmoid() >= 0.5).squeeze(1).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    metrics = calculate_metrics(all_labels, all_preds)

    print("Test set evaluation")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred; class 0=fake, 1=real)")
    print("             Pred Fake  Pred Real")
    cm = metrics["confusion_matrix"]
    if cm.shape == (2, 2):
        print(f"True Fake   {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"True Real   {cm[1][0]:5d}  {cm[1][1]:5d}")
    else:
        print(cm)

    return metrics


if __name__ == "__main__":
    evaluate()
