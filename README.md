# Distributed ASL-to-Speech Translation System

A Windows/VS Code friendly distributed-programming project that recognizes ASL hand input, converts it to text, and speaks the result.

## Architecture

Webcam → MediaPipe landmarks → Desktop UI → FastAPI coordinator → Redis broker → Celery worker(s) → TensorFlow model → result → Text-to-Speech

The system is distributed because the client, coordinator, broker, alphabet worker and phrase worker are separate processes and can be placed on separate machines by changing `REDIS_URL` and the API address.

## AI design

- **Alphabet A-Z:** public ASL image dataset → MediaPipe 21 hand landmarks (63 normalized x/y/z features) → small TensorFlow neural network.
- **Dynamic/common signs:** 30-frame landmark sequences → GRU model.
- **Phrase normalization:** fingerspelled text such as `THANKYOU` can also be normalized to `Thank you`.
- No third-party pretrained ASL model is required.

## 1. Open in VS Code

Extract the project, then `File → Open Folder`.

Recommended Python: **3.12**.

## 2. Create environment / install packages

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

If TensorFlow is the only package that fails:
```powershell
python -m pip install --no-cache-dir tensorflow==2.19.1
python -m pip install --no-cache-dir -r requirements.txt
```

## 3. Prepare the public A-Z dataset

Download an ASL alphabet image dataset whose folders are A, B, C ... Z. Put/copy the class folders into:

`datasets/alphabet_images/`

Expected:
```text
datasets/alphabet_images/
├── A/
├── B/
...
└── Z/
```

Then extract MediaPipe features:
```powershell
python training\extract_alphabet_landmarks.py
```

Train:
```powershell
python training\train_alphabet.py
```

Outputs:
```text
models/alphabet_landmarks.keras
models/alphabet_labels.json
```

## 4. Optional: train common/dynamic signs

Collect 40 examples of each sign:
```powershell
python training\collect_phrase_sequences.py --label HELLO --samples 40
python training\collect_phrase_sequences.py --label THANK_YOU --samples 40
python training\collect_phrase_sequences.py --label PLEASE --samples 40
python training\collect_phrase_sequences.py --label YES --samples 40
python training\collect_phrase_sequences.py --label NO --samples 40
python training\collect_phrase_sequences.py --label HELP --samples 40
```

For a stronger model use more signers and more samples, or preprocess an appropriately licensed public word-level video dataset into 30x63 landmark sequences.

Train:
```powershell
python training\train_phrases.py
```

Outputs:
`models/phrase_sequence.keras` and `models/phrase_labels.json`.

## 5. Start the distributed system

Start Docker Desktop.

**Terminal 1 — Redis**
```powershell
docker compose up -d redis
docker compose ps
```

**Terminal 2 — Coordinator**
```powershell
.\.venv\Scripts\Activate.ps1
.\start_coordinator.bat
```
Test: `http://127.0.0.1:8000/health`

**Terminal 3 — Alphabet worker**
```powershell
.\.venv\Scripts\Activate.ps1
.\start_alphabet_worker.bat
```
The task list MUST show `workers.alphabet_tasks.predict_alphabet`.

**Terminal 4 — Phrase worker**
```powershell
.\.venv\Scripts\Activate.ps1
.\start_phrase_worker.bat
```

**Terminal 5 — UI**
```powershell
.\.venv\Scripts\Activate.ps1
.\start_ui.bat
```

Click **Start Camera**, show a trained letter, build text, then press **Speak**.

## Important: restart workers after training

Celery loads the model when it first receives a task. If you train or replace a `.keras` file while a worker is already running, stop that worker with `Ctrl+C` and start it again.

## Fix included for your previous Celery KeyError

The Celery application now explicitly imports:
`workers.alphabet_tasks` and `workers.phrase_tasks`.

The Windows workers also use `-P solo`. When the alphabet worker starts, confirm its `[tasks]` section contains:
`workers.alphabet_tasks.predict_alphabet`.

## Public dataset suggestions

- ASL Alphabet (static A-Z images): https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- WLASL (word-level ASL video research dataset): https://github.com/dxli94/WLASL

Check each dataset's license/terms before redistribution. This ZIP intentionally does not redistribute those datasets.

## Suggested demonstration

1. Show Docker Redis.
2. Show FastAPI `/health`.
3. Show alphabet and phrase Celery workers in separate VS Code terminals.
4. Start the UI.
5. Sign letters to build a word.
6. Use Smart Phrase if desired.
7. Press Speak.
8. Explain that workers can be moved to other computers while Redis coordinates tasks.
