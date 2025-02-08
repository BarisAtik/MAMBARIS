import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from tqdm import tqdm
from check_and_measure import evaluate_model, save_checkpoint, load_last_checkpoint

if 'COLAB_GPU' in os.environ:
    from google.colab import files  # Only when using google colab GPU's

def train_model(model, train_loader, test_loader, model_name, num_epochs=2000, device='cuda',
                checkpoint_dir=None):
    """Training function optimized for studying overfitting behaviors."""
    
    if checkpoint_dir is None:
        checkpoint_dir = f'{model_name}_checkpoints'
        
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        raise RuntimeError(
            f"Directory {checkpoint_dir} already contains files. Please use an empty directory "
            "or use continue_training() to resume from the last checkpoint."
        )
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Increased learning rate and removed weight decay to encourage overfitting
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    # Modified scheduler with very gradual lr reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.9,  # Very gradual reduction
        patience=200,  # Much longer patience
        verbose=True
    )
    
    # Training stability parameters
    max_grad_norm = 1.0  # Keep this to prevent complete instability
    best_accuracy = 0
    checkpoint_freq = 100  # Save checkpoints every 100 epochs
    
    metrics = {
        'train_losses': [], 'test_losses': [],
        'train_accuracies': [], 'test_accuracies': [],
        'train_confidences': [], 'test_confidences': [],
        'learning_rates': [],
        'epoch_train_confidences': [], 'epoch_test_confidences': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        train_confidences = []
        
        for inputs, labels in train_loader:  # Removed tqdm progress bar
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, probabilities = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Monitor gradients for analysis
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Only print gradient warnings every 100 epochs
            if (epoch + 1) % 100 == 0:
                if grad_norm < 1e-4:
                    print(f"Warning: Very small gradients detected: {grad_norm}")
                elif grad_norm >= max_grad_norm:
                    print(f"Warning: Large gradients detected: {grad_norm}")
            
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            confidence, _ = torch.max(probabilities, 1)
            
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_confidences.extend(confidence.detach().cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / total_samples
        train_avg_confidence = np.mean(train_confidences)
        
        test_loss, test_accuracy, test_avg_confidence, test_confidences = evaluate_model(
            model, test_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_accuracy)
        
        # Update metrics without early stopping
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_model_path)
        
        metrics['train_losses'].append(train_loss)
        metrics['test_losses'].append(test_loss)
        metrics['train_accuracies'].append(train_accuracy)
        metrics['test_accuracies'].append(test_accuracy)
        metrics['train_confidences'].append(train_avg_confidence)
        metrics['test_confidences'].append(test_avg_confidence)
        metrics['learning_rates'].append(current_lr)
        metrics['epoch_train_confidences'].append(train_confidences)
        metrics['epoch_test_confidences'].append(test_confidences)
        
        # Print training progress only every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.6f} | Train Acc: {train_accuracy:.1f}% | Test Acc: {test_accuracy:.1f}% | Gap: {train_accuracy - test_accuracy:.1f}%')  # Monitor overfitting
        
        if (epoch + 1) % checkpoint_freq == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_path)
            
            # Create JSON metrics dictionary
            json_metrics = {
                'train_losses': [float(x) for x in metrics['train_losses']],
                'test_losses': [float(x) for x in metrics['test_losses']],
                'train_accuracies': [float(x) for x in metrics['train_accuracies']],
                'test_accuracies': [float(x) for x in metrics['test_accuracies']],
                'train_confidences': [float(x) for x in metrics['train_confidences']],
                'test_confidences': [float(x) for x in metrics['test_confidences']],
                'learning_rates': [float(x) for x in metrics['learning_rates']],
                'current_epoch': epoch + 1
            }
            
            # Save metrics JSON
            metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(json_metrics, f, indent=4)

            # Download if in Colab
            if 'COLAB_GPU' in os.environ:
                try:
                    files.download(checkpoint_path)
                    files.download(metrics_path)
                    print(f"Downloaded checkpoint and metrics for epoch {epoch+1}")
                except Exception as e:
                    print(f"Warning: Could not download files: {str(e)}")
            else:
                print(f"Saved checkpoint and metrics locally for epoch {epoch+1}")
    
    return metrics

def continue_training(model, train_loader, test_loader, model_name, checkpoint_dir, target_epochs=2000, 
                     device='cuda'):
    """Continue training from last checkpoint while maintaining overfitting conditions."""
    checkpoint, last_epoch = load_last_checkpoint(checkpoint_dir)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Maintain same overfitting-friendly optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.9,
        patience=200,
        verbose=True
    )
    
    if checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    with open(os.path.join(checkpoint_dir, 'training_metrics.json'), 'r') as f:
        metrics = json.load(f)
    
    complete_metrics = {
        'train_losses': metrics['train_losses'],
        'test_losses': metrics['test_losses'],
        'train_accuracies': metrics['train_accuracies'],
        'test_accuracies': metrics['test_accuracies'],
        'train_confidences': metrics['train_confidences'],
        'test_confidences': metrics['test_confidences'],
        'learning_rates': metrics.get('learning_rates', []),
        'epoch_train_confidences': checkpoint['metrics']['epoch_train_confidences'],
        'epoch_test_confidences': checkpoint['metrics']['epoch_test_confidences']
    }
    
    criterion = nn.CrossEntropyLoss()
    max_grad_norm = 1.0
    best_accuracy = max(metrics['test_accuracies'])
    checkpoint_freq = 100
    
    print(f"Continuing training from epoch {last_epoch} to {target_epochs}")
    
    for epoch in range(last_epoch, target_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        train_confidences = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{target_epochs}]'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, probabilities = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Monitor gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if grad_norm < 1e-4:
                print(f"Warning: Very small gradients detected: {grad_norm}")
            elif grad_norm >= max_grad_norm:
                print(f"Warning: Large gradients detected: {grad_norm}")
            
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            confidence, _ = torch.max(probabilities, 1)
            
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_confidences.extend(confidence.detach().cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / total_samples
        train_avg_confidence = np.mean(train_confidences)
        
        test_loss, test_accuracy, test_avg_confidence, test_confidences = evaluate_model(
            model, test_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, complete_metrics, best_model_path)
        
        complete_metrics['train_losses'].append(train_loss)
        complete_metrics['test_losses'].append(test_loss)
        complete_metrics['train_accuracies'].append(train_accuracy)
        complete_metrics['test_accuracies'].append(test_accuracy)
        complete_metrics['train_confidences'].append(train_avg_confidence)
        complete_metrics['test_confidences'].append(test_avg_confidence)
        complete_metrics['learning_rates'].append(current_lr)
        complete_metrics['epoch_train_confidences'].append(train_confidences)
        complete_metrics['epoch_test_confidences'].append(test_confidences)
        
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Confidence: {train_avg_confidence:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, Confidence: {test_avg_confidence:.4f}')
        print(f'Train-Test Accuracy Gap: {train_accuracy - test_accuracy:.2f}%')
        
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, complete_metrics, checkpoint_path)
            
            json_metrics = {
                'train_losses': [float(x) for x in complete_metrics['train_losses']],
                'test_losses': [float(x) for x in complete_metrics['test_losses']],
                'train_accuracies': [float(x) for x in complete_metrics['train_accuracies']],
                'test_accuracies': [float(x) for x in complete_metrics['test_accuracies']],
                'train_confidences': [float(x) for x in complete_metrics['train_confidences']],
                'test_confidences': [float(x) for x in complete_metrics['test_confidences']],
                'learning_rates': [float(x) for x in complete_metrics['learning_rates']],
                'current_epoch': epoch + 1
            }
            with open(os.path.join(checkpoint_dir, 'training_metrics.json'), 'w') as f:
                json.dump(json_metrics, f, indent=4)
    
    return complete_metrics