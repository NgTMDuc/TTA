import torch

# Define parameters
B = 5  # Batch size
C = 3  # Number of classes

# Simulate outputs of shape (B, C)
outputs = torch.randn(B, C)

# Define boolean filters
filter_ids_0 = torch.tensor([True, False, True, False, True])
filter_ids_1 = torch.tensor([True, False, True])

# Perform the operation
prob_outputs = outputs[filter_ids_0][filter_ids_1].softmax(1)

# Check the shapes
print("Shape of outputs:", outputs.shape)
print("Shape after first filter:", outputs[filter_ids_0].shape)
print("Shape of prob_outputs:", prob_outputs.shape)