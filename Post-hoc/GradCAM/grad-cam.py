"""
GradCAM implementation using official pytorch-grad-cam library
Install with: pip install grad-cam
"""

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.io import decode_image
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Official GradCAM imports
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval()
        self.transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
    
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.transforms(x)
            y = self.model(x)
            return y.argmax(dim=1)


def prepare_image_for_gradcam(img_tensor, transforms):
    """
    Prepare image for GradCAM
    Args:
        img_tensor: Original image tensor (C, H, W) with values in [0, 255]
        transforms: Torchvision transforms
    Returns:
        input_tensor: Normalized tensor for model (1, C, H, W)
        rgb_img: Resized image as numpy array (H, W, C) in [0, 1] for visualization
    """
    # Convert RGBA to RGB if necessary
    if img_tensor.shape[0] == 4:
        print("  Note: Image has alpha channel, converting RGBA → RGB")
        # Extract RGB channels only
        img_tensor = img_tensor[:3, :, :]
    
    # Apply model transforms and add batch dimension
    input_tensor = transforms(img_tensor.float()).unsqueeze(0)
    
    # Get the transformed image for visualization (before normalization)
    # We need to resize to 224x224 to match the model input
    from torchvision.transforms.functional import resize
    resized_img = resize(img_tensor.float(), [224, 224], antialias=True)
    
    # Convert to numpy and normalize to [0, 1] for visualization
    rgb_img = resized_img.cpu().permute(1, 2, 0).numpy() / 255.0
    rgb_img = np.clip(rgb_img, 0, 1)
    
    return input_tensor, rgb_img


def visualize_multiple_cams(original_img, cam_dict, prediction, label_text, save_path=None):
    """
    Visualize multiple CAM methods side by side
    Args:
        original_img: Original RGB image (H, W, C) in [0, 1]
        cam_dict: Dictionary of {method_name: cam_image}
        prediction: Predicted class
        label_text: Label text
        save_path: Path to save visualization
    """
    n_methods = len(cam_dict) + 1  # +1 for original image
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    
    if n_methods == 2:
        axes = [axes[0], axes[1]]
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # CAM visualizations
    for idx, (method_name, cam_img) in enumerate(cam_dict.items(), 1):
        axes[idx].imshow(cam_img)
        axes[idx].set_title(f'{method_name}')
        axes[idx].axis('off')
    
    plt.suptitle(f'Prediction: {label_text}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def run_gradcam_analysis(img, img_name, model, transforms, target_layers, labels, device, 
                         methods=['GradCAM', 'GradCAM++', 'XGradCAM', 'EigenCAM']):
    """
    Run multiple GradCAM methods on an image
    Args:
        img: Image tensor (C, H, W)
        img_name: Name for saving files
        model: PyTorch model
        transforms: Image transforms
        target_layers: List of target layers for CAM
        labels: ImageNet labels dictionary
        device: Device to run on
        methods: List of CAM methods to use
    """
    print(f"\n{'='*60}")
    print(f"Processing {img_name}")
    print(f"{'='*60}")
    
    # Prepare image
    input_tensor, rgb_img = prepare_image_for_gradcam(img, transforms)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    label_info = labels[str(pred_class)]
    label_text = f"{label_info[1]} (class {pred_class}, conf: {confidence:.3f})"
    print(f"Prediction: {label_text}")
    
    # Dictionary to store CAM results
    cam_results = {}
    
    # Available CAM methods
    cam_methods = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'XGradCAM': XGradCAM,
        'EigenCAM': EigenCAM,
        'HiResCAM': HiResCAM,
        'ScoreCAM': ScoreCAM,
        'AblationCAM': AblationCAM,
        'FullGrad': FullGrad,
    }
    
    # Target for the predicted class
    targets = [ClassifierOutputTarget(pred_class)]
    
    # Run each CAM method
    for method_name in methods:
        if method_name not in cam_methods:
            print(f"Warning: {method_name} not available, skipping...")
            continue
        
        print(f"\nRunning {method_name}...")
        
        try:
            # Initialize CAM (no use_cuda parameter in newer versions)
            cam_algorithm = cam_methods[method_name](
                model=model,
                target_layers=target_layers
            )
            
            # Generate CAM
            grayscale_cam = cam_algorithm(input_tensor=input_tensor, targets=targets)
            
            # Take the CAM for the first image (we only have one)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay CAM on image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            cam_results[method_name] = cam_image
            
            print(f"  ✓ {method_name} completed")
            print(f"    - CAM range: [{grayscale_cam.min():.3f}, {grayscale_cam.max():.3f}]")
            print(f"    - CAM mean: {grayscale_cam.mean():.3f}")
            
        except Exception as e:
            print(f"  ✗ {method_name} failed: {str(e)}")
    
    # Visualize results
    if cam_results:
        save_path = f'{img_name}_gradcam_comparison.png'
        visualize_multiple_cams(rgb_img, cam_results, pred_class, label_text, save_path)
        print(f"\n✓ Visualization saved to: {save_path}")
    
    return pred_class, label_text, cam_results


# Main execution
if __name__ == "__main__":
    print("GradCAM Analysis with Official pytorch-grad-cam Library")
    print("="*60)
    
    # Load images
    dog1 = decode_image('dog1-removebg-preview.png')
    dog2 = decode_image('dog2-removebg-preview.png')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval().to(device)
    transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
    
    # Target layer for GradCAM (last convolutional layer)
    target_layers = [model.layer4[-1]]
    
    print(f"Target layer: {target_layers[0].__class__.__name__}")
    
    # Load ImageNet labels
    with open('imagenet_class_index.json') as labels_file:
        labels = json.load(labels_file)
    
    # CAM methods to compare
    methods_to_compare = ['GradCAM', 'GradCAM++', 'XGradCAM', 'EigenCAM']
    
    print(f"\nCAM methods to run: {', '.join(methods_to_compare)}")
    
    # Process images
    images = [
        (dog1, 'dog1'),
        (dog2, 'dog2')
    ]
    
    results = []
    for img, name in images:
        result = run_gradcam_analysis(
            img=img,
            img_name=name,
            model=model,
            transforms=transforms,
            target_layers=target_layers,
            labels=labels,
            device=device,
            methods=methods_to_compare
        )
        results.append(result)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    for idx, ((img, name), (pred, label, cams)) in enumerate(zip(images, results), 1):
        print(f"\n{name}:")
        print(f"  Prediction: {label}")
        print(f"  Methods run: {', '.join(cams.keys())}")


"""
AVAILABLE CAM METHODS:
======================

1. GradCAM: Original gradient-weighted class activation mapping
   - Fast and reliable
   - Good baseline method

2. GradCAM++: Improved version with better localization
   - Weights gradients with pixel-wise class scores
   - Better for multiple objects

3. XGradCAM: Weighted gradients by activation values
   - More axiomatically justified
   - Similar to GradCAM but with different weighting

4. EigenCAM: Uses principal components of activations
   - No gradient computation needed
   - Fast and interpretable

5. HiResCAM: High-resolution CAM
   - Better spatial resolution
   - More detailed visualizations

6. ScoreCAM: Gradient-free approach
   - Uses forward passes only
   - Slower but more stable

7. AblationCAM: Removes features to measure importance
   - Most interpretable
   - Computationally expensive

8. FullGrad: Uses full gradients (input + bias)
   - More complete attribution
   - Better handles bias terms

USAGE NOTES:
============
- GradCAM, GradCAM++, XGradCAM: Fast, gradient-based, good for most cases
- EigenCAM: Fast, no gradients, good for quick analysis
- ScoreCAM, AblationCAM: Slower but more stable
- HiResCAM: Best for detailed localization
- FullGrad: Most theoretically complete

For most applications, start with GradCAM and GradCAM++
"""