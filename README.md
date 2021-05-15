# pytorch-fashion-mnist
Trying out Image Classification on FashionMNIST dataset using PyTorch.

## Key concepts:
  1. Datasets and Dataloaders.
  2. Explore the data tensors
  3. Define the network architecture by inherting from `nn.Module` class (2 crucial methods to define are `__init__` and `forward`)
  4. Instantiate model
  5. Define loss function (aka objective function) and optimizer
  6. Define a training method that runs once per epoch
  7. Define a test method that evaluates the performance of the model
  8. Define the training loop that runs for several epochs
  9. Save/Load the model
  10. Use the loaded model for prediction
  11. How to perform the computations on CPUs/GPUs

PyTorch documentation has excellent tutorial about FashionMNIST on their webpage.
