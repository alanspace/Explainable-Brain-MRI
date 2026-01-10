import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, typically (grad_tensor,)
        self.gradients = grad_output[0]

    def forward(self, input_image, class_idx=None, use_pp=True):
        """
        Args:
            input_image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            class_idx (int, optional): Target class index. If None, uses top prediction.
            use_pp (bool): If True, uses Grad-CAM++ (Better localization).
        Returns:
            heatmap (np.array): Visual heatmap of shape (H, W)
            idx (int): The class index explained
        """
        # 1. Forward pass
        self.model.zero_grad()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward pass for the target class
        score = output[0, class_idx]
        score.backward()
        
        # 3. Compute CAM
        gradients = self.gradients # (1, C, H, W)
        activations = self.activations # (1, C, H, W)
        b, c, h, w = gradients.size()
        
        if use_pp:
            # Grad-CAM++: Generalized weights for better object localization
            # Formula: w_ij = (grad_ij)^2 / ( 2*(grad_ij)^2 + sum(A_kl * (grad_kl)^3) )
            
            grads_power_2 = gradients.pow(2)
            grads_power_3 = gradients.pow(3)
            
            # Sum over H, W dimensions
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            
            eps = 1e-7
            # Alpha coefficients
            aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
            
            # Replace NaNs with 0 (where gradients are 0)
            aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))
            
            # Weights: Sum over (i, j) of [ aij * relu(grad_ij) ]
            # Note: The original paper says ReLU(gradients), not gradients.
            weights = torch.sum(aij * torch.relu(gradients), dim=(2, 3), keepdim=True) # (1, C, 1, 1)
            
            # Weighted combination of activations
            # (1, C, 1, 1) * (1, C, H, W) -> (1, C, H, W)
            cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, H, W)
            
        else:
            # Traditional Grad-CAM
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True) # (1, C, 1, 1)
            cam = torch.sum(pooled_gradients * activations, dim=1, keepdim=True)
            
        heatmap = cam.squeeze()
        
        # ReLU extraction
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap = heatmap.detach().cpu().numpy()
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, class_idx

    @staticmethod
    def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlays heatmap onto the original image.
        Args:
            heatmap: (H, W) float [0,1]
            original_image: (H, W, 3) numpy array (uint8) or suitable float
        """
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to RGB colormap
        heatmap_uint8 = (255 * heatmap).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        superimposed_img = heatmap_colored * alpha + original_image * (1 - alpha)
        
        # Normalize for display
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img
