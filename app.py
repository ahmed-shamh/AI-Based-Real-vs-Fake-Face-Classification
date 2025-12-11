from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import timm
from PIL import Image
import io
from torchvision import transforms
import datetime
import os

# ============================
# CREATE APP
# ============================
app = FastAPI(title="Deep Fake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# LOAD XCEPTION MODEL
# ============================
XCEPTION_PATH = "best_model.pth"
xception_model = None

try:
    print(f"🔄 Loading Xception model...")
    checkpoint = torch.load(XCEPTION_PATH, map_location="cpu")
    xception_model = timm.create_model("xception", pretrained=False, num_classes=1)
    xception_model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(xception_model.fc.in_features, 1)
    )
    xception_model.load_state_dict(checkpoint["model_state"])
    xception_model.eval()
    print("✅ Xception model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Xception model: {e}")

# ============================
# LOAD SWIN TRANSFORMER MODEL
# ============================
SWIN_PATH = r"checkpoint_epoch_8.pth"
swin_model = None

if os.path.exists(SWIN_PATH):
    try:
        print(f"🔄 Loading Swin model...")
        checkpoint = torch.load(SWIN_PATH, map_location="cpu")
        
        # Create Swin model
        swin_model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=1)
        
        # Check if we need to modify the head
        num_features = swin_model.head.in_features
        swin_model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1)
        )
        
        # Load state dict
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                swin_model.load_state_dict(checkpoint["model_state"])
            elif "state_dict" in checkpoint:
                swin_model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try to load the entire dict
                swin_model.load_state_dict(checkpoint)
        else:
            swin_model.load_state_dict(checkpoint)
        
        swin_model.eval()
        print("✅ Swin Transformer model loaded successfully")
        
        # Test the model output shape
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            test_output = swin_model(dummy)
            print(f"   Test output shape: {test_output.shape if hasattr(test_output, 'shape') else type(test_output)}")
            if isinstance(test_output, torch.Tensor):
                print(f"   Test output numel: {test_output.numel()}")
            
    except Exception as e:
        print(f"❌ Error loading Swin model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ Swin model file not found at: {SWIN_PATH}")

# ============================
# IMAGE TRANSFORMS
# ============================
# Xception transform
xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Swin transform
swin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================
# ROUTES
# ============================
@app.get("/")
def home():
    return {
        "message": "Deepfake Detection API with Xception and Swin Models",
        "endpoints": {
            "xception_predict": "POST /predict/xception",
            "swin_predict": "POST /predict/swin",
            "health": "GET /health"
        },
        "models_loaded": {
            "xception": xception_model is not None,
            "swin": swin_model is not None
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "xception": xception_model is not None,
            "swin": swin_model is not None
        },
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/predict/xception")
async def predict_xception(file: UploadFile = File(...)):
    """Predict using Xception model only"""
    if xception_model is None:
        return {
            "error": "Xception model not loaded",
            "model": "xception",
            "success": False
        }
    
    try:
        if not file.content_type.startswith('image/'):
            return {"error": "File must be an image", "model": "xception", "success": False}
        
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = xception_transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = xception_model(img_tensor)
            
            # Handle output shape
            if output.dim() > 1:
                output = output.squeeze()
            
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
        
        confidence = prob if pred == 1 else (1 - prob)
        confidence_percentage = round(confidence * 100, 2)
        
        return {
            "model": "xception",
            "prediction": pred,
            "label": "Fake" if pred == 1 else "Real",
            "probability": float(prob),
            "confidence": float(confidence),
            "confidence_percentage": confidence_percentage,
            "confidence_formatted": f"{confidence_percentage}%",
            "file_name": file.filename,
            "file_type": file.content_type,
            "success": True
        }
        
    except Exception as e:
        print(f"Error in Xception prediction: {e}")
        return {
            "error": str(e),
            "model": "xception",
            "success": False
        }

@app.post("/predict/swin")
async def predict_swin(file: UploadFile = File(...)):
    """Predict using Swin Transformer model only"""
    if swin_model is None:
        return {
            "error": "Swin Transformer model not loaded",
            "model": "swin",
            "success": False
        }
    
    try:
        if not file.content_type.startswith('image/'):
            return {"error": "File must be an image", "model": "swin", "success": False}
        
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = swin_transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = swin_model(img_tensor)
            
            # Handle output shape - Swin might output multiple values
            # Ensure we get a single scalar value
            if isinstance(output, tuple):
                output = output[0]
            
            # Debug: print output shape for troubleshooting
            print(f"DEBUG: Swin output shape: {output.shape if hasattr(output, 'shape') else type(output)}, numel: {output.numel() if hasattr(output, 'numel') else 'N/A'}")
            
            # Ensure output is a tensor
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Model output is not a tensor: {type(output)}")
            
            # Flatten the output to handle any shape [batch, ...] or [...]
            output_flat = output.flatten()
            
            # Extract a single scalar value
            # If we have multiple values, take the first one
            # This handles cases where model outputs [batch, features] or [features]
            if output_flat.numel() == 0:
                raise ValueError("Empty output tensor from model")
            elif output_flat.numel() == 1:
                # Single value - extract it
                output_scalar = output_flat[0]
            else:
                # Multiple values - take the first one (shouldn't happen with num_classes=1, but handle it)
                print(f"WARNING: Model output has {output_flat.numel()} elements, expected 1. Using first element.")
                output_scalar = output_flat[0]
            
            # Apply sigmoid and convert to probability
            prob = torch.sigmoid(output_scalar).item()
            pred = 1 if prob > 0.5 else 0
        
        confidence = prob if pred == 1 else (1 - prob)
        confidence_percentage = round(confidence * 100, 2)
        
        return {
            "model": "swin_transformer",
            "prediction": pred,
            "label": "Fake" if pred == 1 else "Real",
            "probability": float(prob),
            "confidence": float(confidence),
            "confidence_percentage": confidence_percentage,
            "confidence_formatted": f"{confidence_percentage}%",
            "file_name": file.filename,
            "file_type": file.content_type,
            "success": True
        }
        
    except Exception as e:
        print(f"Error in Swin prediction: {e}")
        return {
            "error": str(e),
            "model": "swin",
            "success": False
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Default endpoint - uses Xception if available"""
    if xception_model is not None:
        return await predict_xception(file)
    elif swin_model is not None:
        return await predict_swin(file)
    else:
        return {"error": "No models available", "success": False}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Deep Fake Detector API")
    print("="*50)
    print(f"Xception Model: {'✅ LOADED' if xception_model is not None else '❌ NOT LOADED'}")
    print(f"Swin Model: {'✅ LOADED' if swin_model is not None else '❌ NOT LOADED'}")
    print("="*50)
    print("\nStarting server: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)