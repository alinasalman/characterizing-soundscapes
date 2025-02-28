import os
import torch
import numpy as np
import pandas as pd
import librosa

# Add PANNs repo to path (if running outside repo folder)
import sys
sys.path.append(os.path.dirname(__file__))

from models import Cnn14_DecisionLevelMax

# ====== Load Model ======
model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320,
                               mel_bins=64, fmin=50, fmax=14000, classes_num=527)

checkpoint = torch.load('Cnn14_DecisionLevelMax_mAP=0.385.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# ====== Load Class Labels ======
with open('metadata/class_labels_indices.csv', 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    LABELS = [line.strip().split(',')[2] for line in lines]

LABELS = np.array(LABELS)

# ====== Audio Loading Function ======
def load_audio(file_path, sample_rate=32000):
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return waveform

# ====== Run SED for One File ======
def detect_events(file_path, confidence_threshold=0.5):
    audio = load_audio(file_path)
    audio = torch.tensor(audio).float().unsqueeze(0)

    with torch.no_grad():
        output_dict = model(audio)

    framewise_output = output_dict['framewise_output'][0].numpy()  # (T, 527)
    frame_duration = 10.0 / framewise_output.shape[0]  # assuming 10s clip

    detected_events = []

    for class_idx in range(527):
        class_confidences = framewise_output[:, class_idx]
        above_threshold = class_confidences > confidence_threshold

        # Group contiguous frames into events
        start_time = None
        for i, is_present in enumerate(above_threshold):
            if is_present and start_time is None:
                start_time = i * frame_duration
            elif not is_present and start_time is not None:
                end_time = i * frame_duration
                detected_events.append({
                    'label': LABELS[class_idx],
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': class_confidences[i-1]
                })
                start_time = None

        # If the event extends to the end of the clip
        if start_time is not None:
            detected_events.append({
                'label': LABELS[class_idx],
                'start_time': start_time,
                'end_time': 10.0,
                'confidence': class_confidences[-1]
            })

    return detected_events

# ====== Process Folder ======
def process_folder(folder_path, max_files=10):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')][:max_files]

    all_events = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        print(f'Processing {file}...')

        events = detect_events(file_path)
        for event in events:
            all_events.append({
                'filename': file,
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'label': event['label'],
                'confidence': event['confidence']
            })

    return all_events

# ====== Save Results to CSV ======
def save_results(events, output_file='sed_results.csv'):
    df = pd.DataFrame(events)
    df.to_csv(output_file, index=False)
    print(f'Saved results to {output_file}')


# ====== Main Execution ======
if __name__ == '__main__':
    folder = r'C:\Users\alina\Desktop\Capstone Project\segmented_audio'  # Update this path
    events = process_folder(folder, max_files=10)  # First 10 files for testing
    save_results(events)