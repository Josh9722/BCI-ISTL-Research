from tensorflow.keras.callbacks import Callback
import time

class CustomLogger(Callback):
    """ Custom callback for cleaner output formatting and logging to file. """
    
    def __init__(self, log_file):
        super(CustomLogger, self).__init__()
        self.log_file = log_file
        
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("============\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self._tic = time.time()

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self._tic            # seconds for this epoch
        logs = logs or {}

        msg = (
            f"Epoch {epoch+1:02d}/{self.params['epochs']} | "
            f"Loss: {logs.get('loss', 0):.4f} | "
            f"Acc: {logs.get('accuracy', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | "
            f"Val Acc: {logs.get('val_accuracy', 0):.4f} | "
            f"Time: {t:5.2f}s"
        )
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")