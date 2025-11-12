import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.io import decode_image
from torchvision.models import resnet18, ResNet18_Weights


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM
        Args:
            model: The model to visualize
            target_layer: The layer to compute GradCAM on (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap
        Args:
            input_tensor: Input image tensor (C, H, W)
            target_class: Target class index (if None, uses predicted class)
        Returns:
            cam: GradCAM heatmap
            prediction: Predicted class index
        """
        # Forward pass
        output = self.model(input_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on positive influence
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class
    
    def __call__(self, input_tensor, target_class=None):
        return self.generate_cam(input_tensor, target_class)
def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Apply heatmap on original image
    Args:
        org_img: Original image (H, W, C) numpy array
        activation_map: GradCAM heatmap
        colormap: OpenCV colormap
        alpha: Transparency of heatmap overlay
    Returns:
        Blended image with heatmap overlay
    """
    # Resize activation map to match image size
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend heatmap with original image
    superimposed_img = heatmap * alpha + org_img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img


def visualize_gradcam(original_img, cam, prediction, label_text, save_path=None):
    """
    Visualize GradCAM results
    Args:
        original_img: Original image tensor (C, H, W)
        cam: GradCAM heatmap
        prediction: Predicted class
        label_text: Label text for the prediction
        save_path: Path to save the visualization
    """
    # Convert tensor to numpy for visualization
    img_np = original_img.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = np.uint8(255 * img_np)
    
    # Apply colormap
    superimposed = apply_colormap_on_image(img_np, cam, alpha=0.4)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {label_text}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load images
    dog1 = decode_image('dog1.jpg')
    dog2 = decode_image('dog2.jpg')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval().to(device)
    transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
    
    # Initialize GradCAM (targeting the last convolutional layer)
    gradcam = GradCAM(model, model.layer4)
    
    # Load labels
    with open('imagenet_class_index.json') as labels_file:
        labels = json.load(labels_file)
    
    # Process images
    images = [dog1, dog2]
    
    for i, img in enumerate(images):
        print(f"\nProcessing Dog {i+1}...")
        
        # Prepare image
        img_tensor = img.float().to(device)
        img_transformed = transforms(img_tensor)
        
        # Generate GradCAM
        cam, pred_class = gradcam(img_transformed, target_class=None)
        
        # Get label
        label_info = labels[str(pred_class)]
        label_text = f"{label_info[1]} ({pred_class})"
        
        print(f"Dog {i+1} predicted as: {label_text}")
        
        # Visualize
        visualize_gradcam(
            img, 
            cam, 
            pred_class, 
            label_text,
            save_path=f'dog{i+1}_gradcam.png'
        )
    
    print("\nGradCAM visualizations saved!")