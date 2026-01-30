import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 
try:
    actions = np.array(sorted(os.listdir(DATA_PATH)))
except FileNotFoundError:
    print("ERROR: MP_Data folder not found. Run video_converter.py first.")
    exit()

no_sequences = 30 
sequence_length = 30 

# --- LOAD DATA ---
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the .npy file
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data (Train on 95%, Test on 5%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- BUILD MODEL ---
model = Sequential()
# Input shape: 30 frames, 258 keypoints per frame
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- TRAIN ---
print("Training started...")
model.fit(X_train, y_train, epochs=150) # 150 epochs is usually enough for small data

model.save('action.h5')
print("Model saved as 'action.h5'")