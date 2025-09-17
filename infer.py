# Imported libraries 
import torch
from torchvision import models, transforms
from PIL import Image


# Device

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load the trained model

NUM_CLASSES = 26
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("asl_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


# Training 

IMG_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# Prediction function

def predict_asl(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return chr(predicted.item() + ord('A'))


# Example usage

if __name__ == "__main__":
    # Enter name of the image
    test_image = "images/Name_of_Image_to_use"  
    letter = predict_asl(model, test_image)
    print(f"Predicted ASL letter: {letter}")
