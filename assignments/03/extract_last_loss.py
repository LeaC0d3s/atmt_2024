import torch

# Load the best checkpoint
checkpoint = torch.load("bpe_merge10000/checkpoints/checkpoint_best.pt", map_location=torch.device('cpu'))
# Inspect checkpoint keys to see what data is available
print("Checkpoint keys:", checkpoint.keys())

last_epoch = checkpoint.get('epoch') 
best_val_loss = checkpoint.get('best_loss')  # might also be 'valid_loss'

print(f"Last Epoch: {last_epoch}")
print(f"Best Validation Loss: {best_val_loss}") # kinda looks more like the perplexity scores than the loss
