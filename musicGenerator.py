import numpy as np
import pretty_midi
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pickle 

MODEL_PATH = "lstm_music_model.keras"
WINDOW_SIZE = 50
GENERATE_LENGTH = 100

# temperature 샘플링 함수 
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature  # 로그 + softmax 조절
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(preds)), p=preds)

# pitch 시퀀스 생성 함수 (sampling 방식 개선)
def generate_pitch_sequence(seed, model, pitch_to_idx, idx_to_pitch, num_classes, length=100, temperature=1.0):
    result = seed[:]
    current = seed[:]

    for _ in range(length):
        encoded = [pitch_to_idx.get(p, 0) for p in current]
        onehot = np.array([np.eye(num_classes)[idx] for idx in encoded])
        onehot = np.expand_dims(onehot, axis=0)

        pred = model.predict(onehot, verbose=0)[0]

        # sampling 방식 개선 (np.argmax → 확률 기반)
        next_idx = sample_with_temperature(pred, temperature)
        next_pitch = idx_to_pitch[next_idx]

        result.append(next_pitch)
        current = current[1:] + [next_pitch]

    return result

# MIDI 변환 함수
def pitch_seq_to_midi(pitches, output_path="generated.mid", duration=0.5):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    start = 0.0
    for pitch in pitches:
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + duration)
        piano.notes.append(note)
        start += duration

    midi.instruments.append(piano)
    midi.write(output_path)

import random
def generate_improv_seed(length=50):
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C Major
    seed = []
    current = random.choice(scale)  # 시작음 랜덤
    for _ in range(length):
        options = [p for p in scale if abs(p - current) <= 7]
        next_pitch = random.choice(options)
        seed.append(next_pitch)
        current = next_pitch
    return seed

# pitch index 매핑 로드(pickle)
with open("pitch_to_idx.pkl", "rb") as f:
    pitch_to_idx = pickle.load(f)
with open("idx_to_pitch.pkl", "rb") as f:
    idx_to_pitch = pickle.load(f)

num_classes = len(pitch_to_idx)

# 모델 로드
model = load_model(MODEL_PATH)

# 시드 랜덤 생성
seed = generate_improv_seed()
assert len(seed) == WINDOW_SIZE

# 생성 및 저장
generated_pitches = generate_pitch_sequence(
    seed, model, pitch_to_idx, idx_to_pitch, num_classes,
    length=100, temperature=1.0)
pitch_seq_to_midi(generated_pitches, "generated_song.mid")

assert len(seed) == WINDOW_SIZE, "시드 길이는 window_size와 같아야 합니다."

# ========== 실행 ==========
if __name__ == "__main__":
    generated_pitches = generate_pitch_sequence(
    seed, model, pitch_to_idx, idx_to_pitch, num_classes,
    length=100, temperature=1.0) 
    pitch_seq_to_midi(generated_pitches, "generated_song.mid")
    print("음악 생성 완료: generated_song.mid 저장됨")