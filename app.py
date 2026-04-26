from flask import Flask, request, jsonify, send_file, send_from_directory
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile
import uuid
import threading
import time
import webbrowser

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = tempfile.mkdtemp()

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_PT = {
    'C': 'Dó', 'C#': 'Dó#', 'D': 'Ré', 'D#': 'Ré#',
    'E': 'Mi', 'F': 'Fá', 'F#': 'Fá#', 'G': 'Sol',
    'G#': 'Sol#', 'A': 'Lá', 'A#': 'Lá#', 'B': 'Si'
}
SCALE_TYPES = {'major': 'Maior', 'minor': 'Menor'}

def detect_key(y, sr):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_corrs = [np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1] for i in range(12)]
    minor_corrs = [np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1] for i in range(12)]
    best_major = np.argmax(major_corrs)
    best_minor = np.argmax(minor_corrs)
    if major_corrs[best_major] >= minor_corrs[best_minor]:
        key_idx, scale, confidence = best_major, 'major', float(major_corrs[best_major])
    else:
        key_idx, scale, confidence = best_minor, 'minor', float(minor_corrs[best_minor])
    note = NOTE_NAMES[key_idx]
    return {
        'note': note, 'note_pt': NOTE_NAMES_PT[note],
        'scale': scale, 'scale_pt': SCALE_TYPES[scale],
        'confidence': round(max(0, min(1, (confidence + 1) / 2)) * 100, 1),
        'key_index': int(key_idx)
    }

def detect_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.squeeze(tempo)), 1)

def semitones_to_target_key(current_key_idx, target_note):
    target_idx = NOTE_NAMES.index(target_note)
    return (target_idx - current_key_idx + 6) % 12 - 6

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    f = request.files['file']
    ext = os.path.splitext(f.filename)[1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f'upload_{uuid.uuid4()}{ext}')
    f.save(tmp_path)
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        key_info = detect_key(y, sr)
        bpm = detect_bpm(y, sr)
        duration = librosa.get_duration(y=y, sr=sr)
        file_id = str(uuid.uuid4())
        sf.write(os.path.join(UPLOAD_FOLDER, f'{file_id}.wav'), y, sr)
        return jsonify({
            'file_id': file_id, 'filename': f.filename,
            'key': key_info, 'bpm': bpm,
            'duration': round(duration, 1), 'sample_rate': sr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route('/shift', methods=['POST'])
def shift():
    data = request.json
    file_id = data.get('file_id')
    semitones = data.get('semitones', 0)
    target_note = data.get('target_note')
    current_key_idx = data.get('current_key_idx', 0)
    src_path = os.path.join(UPLOAD_FOLDER, f'{file_id}.wav')
    if not os.path.exists(src_path):
        return jsonify({'error': 'Arquivo não encontrado. Re-envie o áudio.'}), 404
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)
        if data.get('mode') == 'key' and target_note:
            semitones = semitones_to_target_key(current_key_idx, target_note)
        if semitones == 0:
            return jsonify({'error': 'Nenhuma mudança de tom selecionada.'}), 400
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        out_id = str(uuid.uuid4())
        sf.write(os.path.join(UPLOAD_FOLDER, f'{out_id}.wav'), y_shifted, sr)
        new_key_idx = (current_key_idx + semitones) % 12
        new_note = NOTE_NAMES[new_key_idx]
        return jsonify({
            'out_id': out_id, 'semitones_applied': semitones,
            'new_note': new_note, 'new_note_pt': NOTE_NAMES_PT[new_note]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    data = request.json
    file_id = data.get('file_id')
    semitones = data.get('semitones', 0)
    src_path = os.path.join(UPLOAD_FOLDER, f'{file_id}.wav')
    if not os.path.exists(src_path):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)
        if semitones != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        out_path = os.path.join(UPLOAD_FOLDER, f'prev_{uuid.uuid4()}.wav')
        sf.write(out_path, y, sr)
        return send_file(out_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<out_id>')
def download(out_id):
    out_id = os.path.basename(out_id)
    fmt = request.args.get('format', 'wav').lower()
    name = ''.join(c for c in request.args.get('name', 'musica_transposta') if c.isalnum() or c in (' ', '-', '_')).strip() or 'musica_transposta'
    wav_path = os.path.join(UPLOAD_FOLDER, f'{out_id}.wav')
    if not os.path.exists(wav_path):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    if fmt == 'mp3':
        mp3_path = os.path.join(UPLOAD_FOLDER, f'{out_id}.mp3')
        if not os.path.exists(mp3_path):
            AudioSegment.from_wav(wav_path).export(mp3_path, format='mp3', bitrate='192k')
        return send_file(mp3_path, as_attachment=True, download_name=f'{name}.mp3')
    return send_file(wav_path, as_attachment=True, download_name=f'{name}.wav')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5055")).start()
    print("\n🎵 TomShift iniciando em http://localhost:5055\n")
    app.run(port=5055, debug=False)