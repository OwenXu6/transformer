import torch
import os
from transformers import Wav2Vec2ForCTC
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class SpeechAccentDataset(Dataset):
    def __init__(self, mfccs, fixed_text_vec, accent_labels):
        self.mfccs = mfccs
        self.fixed_text_vec = fixed_text_vec
        self.accent_labels = accent_labels

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, index):
        return self.mfccs[index], self.fixed_text_vec, self.accent_labels[index]


class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, wav2vec2, num_classes_task1, num_classes_task2):
        super(MultiTaskWav2Vec2, self).__init__()
        self.wav2vec2 = wav2vec2

        encoder_dim = 768  # Directly use the value used when initializing the Wav2Vec2 model

        self.fc1 = nn.Linear(encoder_dim, num_classes_task1)
        self.fc2 = nn.Linear(encoder_dim, num_classes_task2)

    def forward(self, mfcc):
        encoder_output = self.wav2vec2(mfcc).last_hidden_state
        output_task1 = self.fc1(encoder_output)
        output_task2 = self.fc2(encoder_output)

        return output_task1, output_task2


# Load data from .pt files
mfccs = torch.load('mfcc.pt')
mfccs = mfccs.squeeze(1)
accent_labels = torch.load('accent.pt')

# Assuming the fixed_text_vec is a predefined one-hot vector for fixed text
fixed_text_vec = torch.load('one_hot_vec.pt')  # Define your fixed text vector

# Initialize dataset and dataloader
dataset = SpeechAccentDataset(mfccs, fixed_text_vec, accent_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize Wav2Vec2 model
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

# Initialize multi-task model
num_classes_text = len(fixed_text_vec) if isinstance(fixed_text_vec, list) else 1
num_classes_accent = 198  # Total number of different accents
model = MultiTaskWav2Vec2(wav2vec_model, num_classes_text, num_classes_accent)
model = model.to(device)

# Loss functions and optimizer
criterion_text = CrossEntropyLoss()
criterion_accent = CrossEntropyLoss()
optimizer = Adam(model.parameters())


num_epochs = 50
save_dir = './model_save_dir'  # Specify your model saving directory here

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (mfcc, text_vec, accent_label) in enumerate(dataloader):
        mfcc = mfcc.transpose(1, 2).to(device)  
        text_vec = text_vec.to(device)
        accent_label = accent_label.to(device)

        # Forward pass
        output_text, output_accent = model(mfcc)

        # Compute loss
        loss_text = criterion_text(output_text.view(output_text.size(0) * output_text.size(1), -1), text_vec.view(-1))
        loss_accent = criterion_accent(output_accent.view(output_accent.size(0) * output_accent.size(1), -1), accent_label.view(-1))

        # Combine losses
        loss = loss_text + loss_accent

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}')

    # Save model after every epoch
    torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch}.pth')
