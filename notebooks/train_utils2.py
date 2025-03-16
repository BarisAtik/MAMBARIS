def train_stabilized_model(model, train_loader, test_loader, model_name, num_epochs=2000, device='cuda',
                   checkpoint_dir=None):
    """Training function optimized for Mamba stability while maintaining existing metrics tracking."""
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import os
    import json
    from check_and_measure import evaluate_model, save_checkpoint, load_last_checkpoint
    
    if checkpoint_dir is None:
        checkpoint_dir = f'{model_name}_checkpoints'
        
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        raise RuntimeError(
            f"Directory {checkpoint_dir} already contains files. Please use an empty directory "
            "or use continue_training() to resume from the last checkpoint."
        )
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # STABILIZATION: Lower learning rate, small weight decay, adjusted betas
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Reduced from 5e-3
        weight_decay=1e-5,  # Small weight decay for stability
        betas=(0.9, 0.95)  # Modified beta2 for better stability
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # STABILIZATION: Modified scheduler with warmup
    class WarmupCosineScheduler:
        def __init__(self, optimizer, warmup_epochs, max_epochs, max_lr, min_lr, last_epoch=-1):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.last_epoch = last_epoch
            self._step_count = 0
            self.step()  # Initialize LR
            
        def step(self, metrics=None):
            self._step_count += 1
            epoch = self._step_count
            
            if epoch <= self.warmup_epochs:
                # Linear warmup
                lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            
            # Update optimizer learning rates
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            return lr
            
        def get_lr(self):
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
            
        def state_dict(self):
            return {
                'step_count': self._step_count,
                'warmup_epochs': self.warmup_epochs,
                'max_epochs': self.max_epochs,
                'max_lr': self.max_lr,
                'min_lr': self.min_lr
            }
            
        def load_state_dict(self, state_dict):
            self._step_count = state_dict['step_count']
            self.warmup_epochs = state_dict['warmup_epochs']
            self.max_epochs = state_dict['max_epochs']
            self.max_lr = state_dict['max_lr']
            self.min_lr = state_dict['min_lr']
    
    # STABILIZATION: Scheduler with warmup and cosine decay        
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=10,
        max_epochs=num_epochs,
        max_lr=1e-3,  # Peak LR after warmup
        min_lr=1e-6   # Final minimum LR
    )
    
    # STABILIZATION: Reduced gradient clipping threshold
    max_grad_norm = 0.5  # Reduced from 1.0
    best_accuracy = 0
    checkpoint_freq = 100  # Save checkpoints every 100 epochs
    
    metrics = {
        'train_losses': [], 'test_losses': [],
        'train_accuracies': [], 'test_accuracies': [],
        'train_confidences': [], 'test_confidences': [],
        'learning_rates': [],
        'epoch_train_confidences': [], 'epoch_test_confidences': []
    }
    
    # STABILIZATION: Gradient accumulation steps
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        train_confidences = []
        
        # STABILIZATION: Reset gradients at start of epoch
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logits, probabilities = model(inputs)
            loss = criterion(logits, labels)
            
            # STABILIZATION: Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Statistics for tracking (use unscaled loss)
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            confidence, _ = torch.max(probabilities, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_confidences.extend(confidence.detach().cpu().numpy())
            
            # STABILIZATION: Gradient accumulation step
            if (i + 1) % accumulation_steps == 0:
                # Monitor gradients for analysis
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Print warnings only occasionally for readability
                if (epoch + 1) % 100 == 0:
                    if grad_norm < 1e-4:
                        print(f"Warning: Very small gradients detected: {grad_norm}")
                    elif grad_norm >= max_grad_norm:
                        print(f"Warning: Large gradients detected: {grad_norm}")
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining samples if dataset size not divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / total_samples
        train_avg_confidence = np.mean(train_confidences)
        
        # Evaluate on test set
        test_loss, test_accuracy, test_avg_confidence, test_confidences = evaluate_model(
            model, test_loader, criterion, device)
        
        # Update learning rate
        current_lr = scheduler.step()
        
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
        
        # Print training progress only every 3 epochs
        if (epoch + 1) % 3 == 0:
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
                    from google.colab import files
                    files.download(checkpoint_path)
                    files.download(metrics_path)
                    print(f"Downloaded checkpoint and metrics for epoch {epoch+1}")
                except Exception as e:
                    print(f"Warning: Could not download files: {str(e)}")
            else:
                print(f"Saved checkpoint and metrics locally for epoch {epoch+1}")
    
    return metrics


def continue_stabilized_training(model, train_loader, test_loader, model_name, checkpoint_dir, target_epochs=2000, 
                              device='cuda'):
    """Continue training from last checkpoint with stability improvements."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import os
    import json
    from tqdm import tqdm
    from check_and_measure import evaluate_model, save_checkpoint, load_last_checkpoint
    
    # Pass model_name as model_type to load_last_checkpoint
    checkpoint, last_epoch = load_last_checkpoint(checkpoint_dir, model_type=model_name)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # STABILIZATION: Better optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate
        weight_decay=1e-5,  # Small weight decay
        betas=(0.9, 0.95)  # Better beta2 value
    )
    
    # Try to load optimizer state if compatible
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("Warning: Could not load optimizer state. Starting with fresh optimizer.")
    
    # Custom scheduler with warmup
    class WarmupCosineScheduler:
        def __init__(self, optimizer, warmup_epochs, max_epochs, max_lr, min_lr, last_epoch=-1):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.last_epoch = last_epoch
            self._step_count = 0
            self.step()  # Initialize LR
            
        def step(self, metrics=None):
            self._step_count += 1
            epoch = self._step_count
            
            if epoch <= self.warmup_epochs:
                # Linear warmup
                lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            
            # Update optimizer learning rates
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            return lr
            
        def get_lr(self):
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
            
        def state_dict(self):
            return {
                'step_count': self._step_count,
                'warmup_epochs': self.warmup_epochs,
                'max_epochs': self.max_epochs,
                'max_lr': self.max_lr,
                'min_lr': self.min_lr
            }
            
        def load_state_dict(self, state_dict):
            self._step_count = state_dict['step_count']
            self.warmup_epochs = state_dict['warmup_epochs']
            self.max_epochs = state_dict['max_epochs']
            self.max_lr = state_dict['max_lr']
            self.min_lr = state_dict['min_lr']
    
    # Initialize scheduler    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,  # Shorter warmup since we're continuing
        max_epochs=target_epochs,
        max_lr=5e-4,  # Peak LR after warmup
        min_lr=1e-6   # Final minimum LR
    )
    
    # Try to load scheduler state if available
    if checkpoint['scheduler_state_dict']:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Warning: Could not load scheduler state. Starting with fresh scheduler.")
    
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
    
    # STABILIZATION: Lower gradient clipping threshold
    max_grad_norm = 0.5
    best_accuracy = max(metrics['test_accuracies'])
    checkpoint_freq = 100
    
    # STABILIZATION: Gradient accumulation
    accumulation_steps = 4
    
    print(f"Continuing training from epoch {last_epoch} to {target_epochs}")
    
    for epoch in range(last_epoch, target_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        train_confidences = []
        
        # Reset gradient at the start of epoch
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch [{epoch+1}/{target_epochs}]')):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logits, probabilities = model(inputs)
            loss = criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Statistics (using unscaled loss)
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            confidence, _ = torch.max(probabilities, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_confidences.extend(confidence.detach().cpu().numpy())
            
            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0:
                # Monitor gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                if grad_norm < 1e-4:
                    print(f"Warning: Very small gradients detected: {grad_norm}")
                elif grad_norm >= max_grad_norm:
                    print(f"Warning: Large gradients detected: {grad_norm}")
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle any remaining samples
        if len(train_loader) % accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / total_samples
        train_avg_confidence = np.mean(train_confidences)
        
        test_loss, test_accuracy, test_avg_confidence, test_confidences = evaluate_model(
            model, test_loader, criterion, device)
        
        current_lr = scheduler.step()
        
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