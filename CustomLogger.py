from tensorflow.keras.callbacks import Callback

class CustomLogger(Callback):
    """ Custom callback for cleaner output formatting and logging to file. """
    
    def __init__(self, log_file):
        super(CustomLogger, self).__init__()
        self.log_file = log_file
        # Optionally initialize the file, e.g. create it or write a header.
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("============\n")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = (
            f"Epoch {epoch + 1:02d}/{self.params['epochs']} | "
            f"Loss: {logs.get('loss', 0):.4f} | "
            f"Acc: {logs.get('accuracy', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | "
            f"Val Acc: {logs.get('val_accuracy', 0):.4f}"
        )
        
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')