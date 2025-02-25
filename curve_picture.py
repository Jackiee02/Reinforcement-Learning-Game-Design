import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the train.log file
log_file_path = 'train(1).log'  # Make sure the path is correct
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Extract Epoch, score, loss, time spent
epochs = []
scores = []
losses = []
time_spent = []

# Use regular expression to extract relevant data
for line in lines:
    match = re.search(
        r"Epoch\s*:\s*(\d+)\s*\|\s*score\s*:\s*(-?\d+\.\d+)\s*\|\s*loss\s*:\s*(-?\d+\.\d+)\s*\|\s*stage\s*:\s*1\s*\|\s*time spent\s*:\s*(\d+\.\d+)",
        line)
    if match:
        epoch = int(match.group(1))
        score = float(match.group(2))
        loss = float(match.group(3))
        time = float(match.group(4))

        epochs.append(epoch)
        scores.append(score)
        losses.append(loss)
        time_spent.append(time)

# Create DataFrame
data = pd.DataFrame({
    'Epoch': epochs,
    'Score': scores,
    'Loss': losses,
    'Time Spent': time_spent
})

# Plot score vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Score'], label='Score over Epochs', color='b')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Score vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the image
plt.savefig('score_vs_epoch.png')
plt.show()

# Plot loss vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Loss'], label='Loss over Epochs', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the image
plt.savefig('loss_vs_epoch.png')
plt.show()

# Plot time spent vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(data['Epoch'], data['Time Spent'], label='Time Spent over Epochs', color='g')
plt.xlabel('Epoch')
plt.ylabel('Time Spent (seconds)')
plt.title('Time Spent vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the image
plt.savefig('time_spent_vs_epoch.png')
plt.show()
