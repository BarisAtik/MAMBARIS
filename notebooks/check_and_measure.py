import torch
import os
import numpy as np

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save model checkpoint and metrics."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

def load_last_checkpoint(checkpoint_dir):
   """Load the last saved checkpoint from specified directory."""
   # Find the latest checkpoint
   checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
   if not checkpoints:
       raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
       
   # Extract epoch numbers and find max
   epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoints]
   last_epoch = max(epochs)
   
   checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{last_epoch}.pt')
   checkpoint = torch.load(checkpoint_path)
   
   return checkpoint, last_epoch

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, probabilities = model(inputs)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            confidence, _ = torch.max(probabilities, 1)
            
            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            confidences.extend(confidence.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    avg_confidence = np.mean(confidences)
    
    return avg_loss, accuracy, avg_confidence, confidences