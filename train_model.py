import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 1. pitch 시퀀스 추출 함수
def extract_pitch_sequences(root_dir, max_files=3000):
    pitch_seq = []
    file_count = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".mid") and not fname.endswith(".midi"):
                continue

            if file_count >= max_files:
                break  # 파일 개수 제한

            midi_path = os.path.join(root, fname)
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
                for instrument in midi.instruments:
                    if instrument.is_drum:
                        continue
                    for note in instrument.notes:
                        pitch_seq.append(note.pitch)
                file_count += 1
            except Exception as e:
                # print(f"Error in {fname}: {e}")
                continue

    return pitch_seq


# 2. 시퀀스 자르기
def make_sequences(pitches, window_size=50):
    X = []
    y = []
    for i in range(len(pitches) - window_size):
        seq_in = pitches[i:i+window_size]
        seq_out = pitches[i+window_size]
        X.append(seq_in)
        y.append(seq_out)
    return X, y

# 3. pitch 시퀀스를 MIDI로 변환
def pitch_seq_to_midi(pitches, output_path="generated.mid"):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    start = 0.0
    duration = 0.5  # 각 음의 지속 시간 (초 단위)
    for pitch in pitches:
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + duration)
        piano.notes.append(note)
        start += duration

    midi.instruments.append(piano)
    midi.write(output_path)

root_dir = "./lakh-midi-clean" 

# 4. 실제 pitch 추출 및 시퀀스 자르기
pitches = extract_pitch_sequences(root_dir)
window_size = 50
X, y = make_sequences(pitches, window_size)

#시퀀스 수 제한
X = X[:3000]
y = y[:3000]

# 5. 인코딩
unique_pitches = sorted(set(pitches))
pitch_to_idx = {p: i for i, p in enumerate(unique_pitches)}
idx_to_pitch = {i: p for p, i in pitch_to_idx.items()}

encoded_X = [[pitch_to_idx[p] for p in seq] for seq in X]
encoded_y = [pitch_to_idx[p] for p in y]

# 6. 원핫 인코딩
num_classes = len(pitch_to_idx)
X_onehot = np.array([to_categorical(seq, num_classes=num_classes) for seq in encoded_X])
y_onehot = to_categorical(encoded_y, num_classes=num_classes)

# 7. 모델 정의
model = Sequential([
    LSTM(128, input_shape=(window_size, num_classes)),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X_onehot, y_onehot, test_size=0.2, random_state=42)

early_stop = EarlyStopping(
    monitor='val_loss',     
    patience=3,             # 3 epoch 연속 개선 없으면 종료
    restore_best_weights=True  
)

# 모델 학습 + 로그 저장
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,              
    batch_size=64,
    callbacks=[early_stop]  
)

# 8. 평가
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=200)
plt.show()

# 예측 샘플
sample_idx = np.random.randint(0, len(X_val))
input_seq = X_val[sample_idx:sample_idx+1]
true_label = np.argmax(y_val[sample_idx])
pred_label = np.argmax(model.predict(input_seq))

print(f"예측 pitch: {idx_to_pitch[pred_label]}, 실제 pitch: {idx_to_pitch[true_label]}")

# 예측
pred = model.predict(input_seq)
pred_label = np.argmax(pred)

print(f"예측 pitch: {idx_to_pitch[pred_label]}, 실제 pitch: {idx_to_pitch[true_label]}")

# 9. 저장
model.save("lstm_music_model.keras")

import pickle
with open("pitch_to_idx.pkl", "wb") as f:
    pickle.dump(pitch_to_idx, f)
with open("idx_to_pitch.pkl", "wb") as f:
    pickle.dump(idx_to_pitch, f)
    import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)