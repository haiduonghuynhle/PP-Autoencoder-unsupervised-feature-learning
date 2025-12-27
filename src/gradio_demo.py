#!/usr/bin/env python3
"""
Gradio Demo: CIFAR-10 Image Classification
CSC14120 - Parallel Programming

This demo uses:
1. Autoencoder encoder for feature extraction
2. Trained SVM classifier for prediction

Usage:
    python gradio_demo.py --autoencoder models/autoencoder_gpu.bin \
                          --svm models/svm_cuml_model_gpu.pkl
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import pickle
import argparse
from PIL import Image

# ============================================================================
# CIFAR-10 Class Names
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Class descriptions for display
CLASS_DESCRIPTIONS = {
    'airplane': '✈️ Airplane - Fixed-wing aircraft',
    'automobile': '🚗 Automobile - Passenger car',
    'bird': '🐦 Bird - Flying animal',
    'cat': '🐱 Cat - Domestic feline',
    'deer': '🦌 Deer - Wild mammal',
    'dog': '🐕 Dog - Domestic canine',
    'frog': '🐸 Frog - Amphibian',
    'horse': '🐴 Horse - Large mammal',
    'ship': '🚢 Ship - Water vessel',
    'truck': '🚛 Truck - Heavy vehicle'
}

# ============================================================================
# Autoencoder Model (Same architecture as C++)
# ============================================================================

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder (not used for classification, but needed to load weights)
        self.dec_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dec_conv3 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def encode(self, x):
        """Extract features (8192-dim)"""
        x = F.relu(self.enc_conv1(x))  # [B, 256, 32, 32]
        x = self.pool(x)                # [B, 256, 16, 16]
        x = F.relu(self.enc_conv2(x))  # [B, 128, 16, 16]
        x = self.pool(x)                # [B, 128, 8, 8]
        return x.view(x.size(0), -1)    # [B, 8192]

    def decode(self, x):
        x = x.view(x.size(0), 128, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def load_autoencoder_weights(model, filepath):
    """Load weights from C++ binary file"""
    with open(filepath, 'rb') as f:
        # Read header
        magic = struct.unpack('i', f.read(4))[0]
        version = struct.unpack('i', f.read(4))[0]
        
        if magic != 0xAE2024:
            raise ValueError(f"Invalid model file format (magic: {hex(magic)})")
        
        # Encoder conv1: [256, 3, 3, 3]
        enc_conv1_w = np.frombuffer(f.read(256*3*3*3*4), dtype=np.float32).reshape(256, 3, 3, 3)
        enc_conv1_b = np.frombuffer(f.read(256*4), dtype=np.float32)
        
        # Encoder conv2: [128, 256, 3, 3]
        enc_conv2_w = np.frombuffer(f.read(128*256*3*3*4), dtype=np.float32).reshape(128, 256, 3, 3)
        enc_conv2_b = np.frombuffer(f.read(128*4), dtype=np.float32)
        
        # Decoder conv1: [128, 128, 3, 3]
        dec_conv1_w = np.frombuffer(f.read(128*128*3*3*4), dtype=np.float32).reshape(128, 128, 3, 3)
        dec_conv1_b = np.frombuffer(f.read(128*4), dtype=np.float32)
        
        # Decoder conv2: [256, 128, 3, 3]
        dec_conv2_w = np.frombuffer(f.read(256*128*3*3*4), dtype=np.float32).reshape(256, 128, 3, 3)
        dec_conv2_b = np.frombuffer(f.read(256*4), dtype=np.float32)
        
        # Decoder conv3: [3, 256, 3, 3]
        dec_conv3_w = np.frombuffer(f.read(3*256*3*3*4), dtype=np.float32).reshape(3, 256, 3, 3)
        dec_conv3_b = np.frombuffer(f.read(3*4), dtype=np.float32)
    
    # Load into model
    model.enc_conv1.weight.data = torch.from_numpy(enc_conv1_w.copy())
    model.enc_conv1.bias.data = torch.from_numpy(enc_conv1_b.copy())
    model.enc_conv2.weight.data = torch.from_numpy(enc_conv2_w.copy())
    model.enc_conv2.bias.data = torch.from_numpy(enc_conv2_b.copy())
    model.dec_conv1.weight.data = torch.from_numpy(dec_conv1_w.copy())
    model.dec_conv1.bias.data = torch.from_numpy(dec_conv1_b.copy())
    model.dec_conv2.weight.data = torch.from_numpy(dec_conv2_w.copy())
    model.dec_conv2.bias.data = torch.from_numpy(dec_conv2_b.copy())
    model.dec_conv3.weight.data = torch.from_numpy(dec_conv3_w.copy())
    model.dec_conv3.bias.data = torch.from_numpy(dec_conv3_b.copy())
    
    return model


# ============================================================================
# Global Models (loaded once at startup)
# ============================================================================

autoencoder = None
svm_model = None
device = None


def load_models(autoencoder_path, svm_path):
    """Load both models"""
    global autoencoder, svm_model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load autoencoder
    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder = Autoencoder()
    autoencoder = load_autoencoder_weights(autoencoder, autoencoder_path)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    print("Autoencoder loaded!")
    
    # Load SVM
    print(f"Loading SVM from {svm_path}...")
    try:
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        print("SVM loaded!")
    except EOFError:
        print(f"ERROR: SVM file '{svm_path}' is corrupted or incomplete!")
        print("Please re-train the SVM model using:")
        print("  python train_svm_cuml.py --train models/train_features_gpu.bin --test models/test_features_gpu.bin --data /content/data/cifar-10-batches-bin")
        raise
    except Exception as e:
        print(f"ERROR loading SVM: {e}")
        raise
    
    return True


# ============================================================================
# Prediction Function
# ============================================================================

def preprocess_image(image):
    """Preprocess image for the model"""
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize to 32x32
    image = image.resize((32, 32), Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Convert to CHW format
    img_array = img_array.transpose(2, 0, 1)
    
    return img_array


def classify_image(image):
    """Classify an input image"""
    global autoencoder, svm_model, device
    
    if autoencoder is None or svm_model is None:
        return "Error: Models not loaded!", None, None
    
    if image is None:
        return "Please upload an image!", None, None
    
    try:
        # Preprocess
        img_array = preprocess_image(image)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = autoencoder.encode(img_tensor)
            reconstructed, _ = autoencoder(img_tensor)
        
        features_np = features.cpu().numpy()
        
        # Predict with SVM
        try:
            # Try cuML predict (returns cupy array)
            import cupy as cp
            features_gpu = cp.asarray(features_np)
            prediction = svm_model.predict(features_gpu)
            predicted_class = int(cp.asnumpy(prediction)[0])
        except:
            # Fallback to sklearn predict
            prediction = svm_model.predict(features_np)
            predicted_class = int(prediction[0])
        
        # Get class name
        class_name = CIFAR10_CLASSES[predicted_class]
        description = CLASS_DESCRIPTIONS[class_name]
        
        # Create confidence display (SVM doesn't have probabilities by default)
        result_text = f"""
## Prediction: {description}

### Details:
- **Class ID:** {predicted_class}
- **Class Name:** {class_name}
- **Feature Vector Size:** 8,192 dimensions

### Model Info:
- Autoencoder: Convolutional (2 encoder layers)
- Classifier: SVM with RBF kernel
"""
        
        # Get reconstructed image for display
        recon_img = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
        recon_img = np.clip(recon_img, 0, 1)
        
        # Resize original for display
        if isinstance(image, np.ndarray):
            display_image = Image.fromarray(image)
        else:
            display_image = image
        display_image = display_image.resize((128, 128), Image.NEAREST)
        
        return result_text, np.array(display_image), (recon_img * 255).astype(np.uint8)
        
    except Exception as e:
        return f"Error during classification: {str(e)}", None, None


# ============================================================================
# Gradio Interface
# ============================================================================

def create_demo():
    """Create the Gradio demo interface"""
    
    with gr.Blocks(title="CIFAR-10 Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🖼️ CIFAR-10 Image Classifier
## CSC14120 - Parallel Programming Final Project

This demo uses a **CUDA-accelerated Autoencoder** for feature extraction and an **SVM classifier** for prediction.

### How it works:
1. Upload an image (any size)
2. The image is resized to 32×32 (CIFAR-10 format)
3. Autoencoder extracts 8,192 features
4. SVM predicts the class

### Supported Classes:
✈️ airplane | 🚗 automobile | 🐦 bird | 🐱 cat | 🦌 deer | 🐕 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚛 truck
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=300
                )
                classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                result_text = gr.Markdown(label="Prediction Result")
                
                with gr.Row():
                    processed_image = gr.Image(
                        label="Processed Input (128×128 preview)",
                        height=150
                    )
                    reconstructed_image = gr.Image(
                        label="Autoencoder Reconstruction",
                        height=150
                    )
        
        # Footer
        gr.Markdown("""
---
**University of Science - Vietnam National University, Ho Chi Minh City**

Built with: CUDA C++ | PyTorch | cuML | Gradio
        """)
        
        # Connect the button
        classify_btn.click(
            fn=classify_image,
            inputs=[input_image],
            outputs=[result_text, processed_image, reconstructed_image]
        )
        
        # Also classify on image upload
        input_image.change(
            fn=classify_image,
            inputs=[input_image],
            outputs=[result_text, processed_image, reconstructed_image]
        )
    
    return demo


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classifier Demo')
    parser.add_argument('--autoencoder', type=str, default='models/autoencoder_gpu.bin',
                        help='Path to autoencoder weights (.bin)')
    parser.add_argument('--svm', type=str, default='models/svm_cuml_model_gpu.pkl',
                        help='Path to SVM model (.pkl)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public shareable link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the server on')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIFAR-10 Image Classifier Demo")
    print("CSC14120 - Parallel Programming")
    print("=" * 60)
    
    # Load models
    try:
        load_models(args.autoencoder, args.svm)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nMake sure you have trained models:")
        print("  1. Autoencoder: models/autoencoder_gpu.bin")
        print("  2. SVM: models/svm_cuml_model_gpu.pkl")
        return
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main()
