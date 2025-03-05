import os
import torch
import numpy as np
import pandas as pd
import librosa

from models import Cnn14

# ===== Load Model =====
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)

checkpoint = torch.load('Cnn14_mAP=0.431.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# ===== Load Labels =====
with open('metadata/class_labels_indices.csv', 'r') as f:
    lines = f.readlines()[1:]
    LABELS = [line.strip().split(',')[2] for line in lines]
LABELS = np.array(LABELS)

# ===== Audio Loading Function =====
def load_audio(file_path, sample_rate=32000):
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return waveform

# ===== Inference Function =====
def classify_audio(file_path):
    audio = load_audio(file_path)
    audio = torch.tensor(audio).float().unsqueeze(0)

    with torch.no_grad():
        prediction = model(audio)['clipwise_output']


    print(f"Prediction type: {type(prediction)}")

    top_indices = prediction.squeeze().argsort(descending=True)[:5]
    return [(LABELS[idx], prediction[0, idx].item()) for idx in top_indices]

# ===== Process Folder =====
def process_folder(folder_path, max_files=None):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')] #[:max_files]

    results = []
    for idx, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        print(f'Processing {file}...')

        top_labels = classify_audio(file_path)
        results.append({'filename': file, 'top_labels': top_labels})
        if idx % 10 == 0:
            pd.DataFrame(results).to_csv("checkpoint.csv", index=False)

    return results

# ===== Save Results =====
def save_results(results, output_file='classification_results.csv'):
    rows = []
    for result in results:
        row = {'filename': result['filename']}
        for i, (label, score) in enumerate(result['top_labels']):
            row[f'label_{i+1}'] = label
            row[f'score_{i+1}'] = score
        rows.append(row)
        

    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f'Saved to {output_file}')

# ===== Main Run =====
if __name__ == '__main__':
    folder = r'C:\Users\alina\Desktop\Capstone Project\segmented_audio'
    results = process_folder(folder)  # First 10 for test
    save_results(results)
