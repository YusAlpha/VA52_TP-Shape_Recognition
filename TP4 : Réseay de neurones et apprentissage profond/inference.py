import torch
import os
import random
import cv2
from torchvision import transforms
from nn_intro2 import myNet  # Import your network class

# Load random test image
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "dataset")
test_dir = os.path.join(data_dir, "test/X")
random_folder = random.choice(os.listdir(test_dir))
final_folder = os.path.join(test_dir, str(random_folder))
# Load random test image
random_img = random.choice(os.listdir(test_dir))

# Load model
net = torch.load("W_CNN.pt")
net.eval()

image = cv2.imread(os.path.join(test_dir, random_img))

if image is not None: 
# Preprocess image without using PIL
  transformation = transforms.Compose([
      transforms.ToPILImage(),  # Convert ndarray to PIL Image
      transforms.Resize([30, 30]),
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()])


  image = transformation(image)
  image = image.unsqueeze(0)

  # Predict
  prediction = net(image)

  # Display image and prediction
  cv2.imshow("Image", cv2.imread(os.path.join(test_dir, random_img)))
  print("Prediction:", prediction.argmax().item())

  cv2.waitKey(0)
  cv2.destroyAllWindows()

else:
  print("Image not found")
