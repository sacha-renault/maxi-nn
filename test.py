import torch

input = torch.Tensor([
    [0.2, 1 ,0.2],
    [0.5, 0.5, 0.5],
]).requires_grad_(True)

y = torch.Tensor([
    [0, 1 ,0],
    [0, 0, 1],
]).requires_grad_(True)

soft_output = torch.softmax(input, 1)
print(soft_output.detach().numpy())
loss = torch.nn.CrossEntropyLoss()(soft_output, y)
loss.backward()


# Print the loss value
print("Loss:", loss.item())

# Print gradients for the input tensor
print("Gradients for input tensor:")
print(input.grad)