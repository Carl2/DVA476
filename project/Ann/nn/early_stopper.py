#!/usr/bin/env python



class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Call this after each epoch with validation loss
        Returns: True if should stop training
        """
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
            self.save_checkpoint(model)
            if self.verbose:
                print(f"📌 Initial best val_loss: {val_loss:.4f}")

        elif val_loss > self.best_loss - self.min_delta:
            # No improvement (or improvement too small)
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered!")

        else:
            # Improvement!
            if self.verbose:
                print(f"✓ Val loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        return self.early_stop

    def save_checkpoint(self, model):
        """Save model state"""
        self.best_model_state = model.state_dict().copy()

    def load_best_model(self, model):
        """Restore best model"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"✓ Restored best model (val_loss: {self.best_loss:.4f})")
