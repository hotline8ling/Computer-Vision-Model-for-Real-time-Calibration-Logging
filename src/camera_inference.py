import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import DigitRecognitionModel

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment digits
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return gray, thresh, contours

def get_digit_regions(frame, contours):
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Increase minimum size threshold to filter out smaller regions
        if w > 30 and h > 30:  # Increased from 20 to 30
            # Calculate contour area and aspect ratio as additional filters
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h
            
            # Filter based on area and aspect ratio
            if area > 900 and 0.2 < aspect_ratio < 2.0:  # Minimum area of 30x30 pixels
                # Extract the region
                region = frame[y:y+h, x:x+w]
                regions.append((region, (x, y, w, h)))
    return regions

def main():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitRecognitionModel().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set confidence threshold
    CONFIDENCE_THRESHOLD = 0.7
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Preprocess the frame
        gray, thresh, contours = preprocess_frame(frame)
        
        # Get digit regions
        regions = get_digit_regions(gray, contours)
        
        # Process each region
        for region, (x, y, w, h) in regions:
            # Convert to PIL Image
            pil_image = Image.fromarray(region)
            
            # Preprocess for model
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Get prediction and confidence
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.exp(output)  # Convert log_softmax to probabilities
                confidence, prediction = torch.max(probabilities, dim=1)
                
                # Only show predictions with high confidence
                if confidence.item() > CONFIDENCE_THRESHOLD:
                    # Draw rectangle and prediction
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{prediction.item()} ({confidence.item():.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Digit Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 