import torch

# This is the "Safety First" line for cloud hosting
device = torch.device('cpu')

# Load your model mapping it to CPU
model = MyModel() 
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
