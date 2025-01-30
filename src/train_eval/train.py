import torch

def train(model, train_loader, val_loader, train_config, model_file_path):
    
    num_epochs, patience, device, criterion, optimizer = train_config.values()
    
    best_val_loss = float('inf')
    patience_counter = 0

    training_losses = []
    validation_losses = []


    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        training_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                # Forward pass
                out = model(data)
                loss = criterion(out, data.y)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        validation_losses.append(avg_val_loss)

        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_file_path)
            print("Best model updated based on validation loss.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


    return training_losses, validation_losses
