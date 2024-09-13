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
# print(soft_output.detach().numpy())

log_pred = torch.log(soft_output + 1e-10) * y
bloss = -torch.sum(log_pred, dim=1)
loss = bloss.mean()

soft_output.retain_grad()
log_pred.retain_grad()
bloss.retain_grad()
loss.retain_grad()
loss.backward()


# Print the loss value
print("loss")
print(loss.detach().numpy())
print(loss.grad)

print("bloss")
print(bloss.detach().numpy())
print(bloss.grad)

print("log_pred")
print(log_pred.detach().numpy())
print(log_pred.grad)

print("soft_output")
print(soft_output.detach().numpy())
print(soft_output.grad)

print("input")
print(input.detach().numpy())
print(input.grad)
