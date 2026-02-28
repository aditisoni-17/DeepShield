import torch
import cv2
import numpy as np

def generate_heatmap(model, image_tensor):

    image_tensor.requires_grad = True
    output = model(image_tensor)

    output.backward()

    gradients = image_tensor.grad.data
    heatmap = gradients.mean(dim=1).squeeze().cpu().numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap