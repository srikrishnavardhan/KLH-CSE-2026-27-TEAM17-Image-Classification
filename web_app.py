import base64
import json
import pickle
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
MODEL_PATH = PROJECT_ROOT / 'models' / 'mixup_cnn.h5'
TEST_BATCH_PATH = PROJECT_ROOT / 'cifar-10-python' / 'cifar-10-batches-py' / 'test_batch'

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mixup test set explorer</title>
  <style>
    :root {
      --ink: #17211b;
      --muted: #68736d;
      --paper: #f5f1e8;
      --panel: #fffdf8;
      --line: #ded8cb;
      --accent: #d95f39;
      --accent-dark: #9e3e25;
      --mint: #2f7d64;
      --mint-soft: #dcefe5;
      --rose-soft: #f7ddd2;
      --shadow: 0 22px 55px rgba(38, 41, 33, .12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Georgia, 'Times New Roman', serif;
      min-height: 100vh;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .32;
      background-image: radial-gradient(#c6bba8 0.7px, transparent 0.7px);
      background-size: 13px 13px;
      mask-image: linear-gradient(to bottom, black, transparent 85%);
    }
    .shell { width: min(1080px, calc(100% - 40px)); margin: 0 auto; padding: 44px 0 64px; position: relative; }
    header { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-bottom: 34px; }
    .eyebrow { color: var(--accent-dark); font: 700 12px/1.2 'Trebuchet MS', sans-serif; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 8px 0 0; font-size: clamp(38px, 6vw, 70px); line-height: .95; letter-spacing: -0.03em; font-weight: 700; max-width: 650px; }
    .dek { color: var(--muted); font: 15px/1.55 'Trebuchet MS', sans-serif; max-width: 330px; margin: 0 0 3px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(310px, .85fr); gap: 24px; align-items: stretch; }
    .image-panel, .details { background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }
    .image-panel { padding: 18px; position: relative; }
    .image-frame { aspect-ratio: 1; background: #25302b; display: grid; place-items: center; overflow: hidden; }
    #sample-image { image-rendering: pixelated; width: 100%; height: 100%; object-fit: contain; transition: opacity .2s ease, transform .2s ease; }
    .image-panel.loading #sample-image { opacity: .35; transform: scale(.98); }
    .caption { display: flex; justify-content: space-between; gap: 16px; padding: 14px 2px 0; color: var(--muted); font: 12px 'Trebuchet MS', sans-serif; }
    .details { padding: 30px; display: flex; flex-direction: column; }
    .section-label { color: var(--muted); font: 700 11px 'Trebuchet MS', sans-serif; letter-spacing: .13em; text-transform: uppercase; }
    .truth { margin-top: 9px; font-size: 38px; line-height: 1; }
    .result { margin: 28px 0 24px; padding: 18px; border-left: 5px solid var(--mint); background: var(--mint-soft); }
    .result.wrong { border-color: var(--accent); background: var(--rose-soft); }
    .result-title { font: 700 22px 'Trebuchet MS', sans-serif; }
    .result-sub { margin-top: 7px; color: var(--muted); font: 14px/1.4 'Trebuchet MS', sans-serif; }
    .score { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; border-top: 1px solid var(--line); padding-top: 18px; }
    .score strong { color: var(--accent-dark); font-size: 34px; }
    .score span { color: var(--muted); font: 12px 'Trebuchet MS', sans-serif; text-align: right; }
    button { border: 0; background: var(--ink); color: #fffdf8; padding: 15px 18px; margin-top: auto; cursor: pointer; font: 700 13px 'Trebuchet MS', sans-serif; letter-spacing: .03em; transition: background .2s ease, transform .2s ease; }
    button:hover { background: var(--accent); transform: translateY(-2px); }
    button:disabled { opacity: .55; cursor: wait; transform: none; }
    .top-predictions { margin: 28px 0; }
    .prediction-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; margin-top: 13px; font: 13px 'Trebuchet MS', sans-serif; }
    .bar { grid-column: 1 / -1; height: 7px; background: #ebe5da; overflow: hidden; }
    .bar i { display: block; height: 100%; background: var(--accent); }
    .footer-note { margin-top: 18px; color: var(--muted); font: 12px/1.5 'Trebuchet MS', sans-serif; }
    @media (max-width: 760px) {
      .shell { width: min(100% - 24px, 560px); padding-top: 28px; }
      header { display: block; }
      .dek { margin-top: 18px; }
      .layout { grid-template-columns: 1fr; }
      .details { min-height: 430px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div><div class="eyebrow">CIFAR-10 / Mixup CNN</div><h1>What did it see?</h1></div>
      <p class="dek">Draw a random image from the untouched test set and inspect the model's call.</p>
    </header>
    <section class="layout" aria-live="polite">
      <article class="image-panel" id="image-panel">
        <div class="image-frame"><img id="sample-image" alt="Random CIFAR-10 test sample"></div>
        <div class="caption"><span id="sample-number">Selecting a sample...</span><span>32 × 32 pixels</span></div>
      </article>
      <article class="details">
        <div class="section-label">Ground truth</div>
        <div class="truth" id="truth">—</div>
        <div class="result" id="result">
          <div class="result-title" id="result-title">Waiting for a sample</div>
          <div class="result-sub" id="result-sub">The model's prediction will appear here.</div>
        </div>
        <div class="score"><strong id="confidence">—</strong><span>model confidence<br>for its prediction</span></div>
        <div class="top-predictions"><div class="section-label">Top predictions</div><div id="predictions"></div></div>
        <button id="next-button" type="button">Draw another test image <span aria-hidden="true">↗</span></button>
        <div class="footer-note">Model: `models/mixup_cnn.h5` · Test set: 10,000 images</div>
      </article>
    </section>
  </main>
  <script>
    const imagePanel = document.getElementById('image-panel');
    const image = document.getElementById('sample-image');
    const sampleNumber = document.getElementById('sample-number');
    const truth = document.getElementById('truth');
    const result = document.getElementById('result');
    const resultTitle = document.getElementById('result-title');
    const resultSub = document.getElementById('result-sub');
    const confidence = document.getElementById('confidence');
    const predictions = document.getElementById('predictions');
    const nextButton = document.getElementById('next-button');

    async function drawSample() {
      nextButton.disabled = true;
      imagePanel.classList.add('loading');
      try {
        const response = await fetch('/api/random-test-image');
        if (!response.ok) throw new Error('Request failed');
        const data = await response.json();
        image.src = data.image;
        sampleNumber.textContent = `Test sample #${data.index + 1} of ${data.dataset_size}`;
        truth.textContent = data.actual;
        result.classList.toggle('wrong', !data.correct);
        resultTitle.textContent = data.correct ? 'Correct prediction' : 'Missed this one';
        resultSub.textContent = `The model called it ${data.predicted}.`;
        confidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        predictions.innerHTML = data.top_predictions.map(item => `
          <div class="prediction-row"><span>${item.label}</span><strong>${(item.confidence * 100).toFixed(1)}%</strong><div class="bar"><i style="width:${item.confidence * 100}%"></i></div></div>
        `).join('');
      } catch (error) {
        resultTitle.textContent = 'Could not load a sample';
        resultSub.textContent = 'Check that the local Python server is running.';
      } finally {
        imagePanel.classList.remove('loading');
        nextButton.disabled = false;
      }
    }

    nextButton.addEventListener('click', drawSample);
    drawSample();
  </script>
</body>
</html>'''


def load_test_data():
    with TEST_BATCH_PATH.open('rb') as file:
        batch = pickle.load(file, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b'labels'])
    return images.astype('float32') / 255.0, labels


print('Loading Mixup model and CIFAR-10 test set...')
MODEL = tf.keras.models.load_model(MODEL_PATH)
TEST_IMAGES, TEST_LABELS = load_test_data()
print('Web app ready.')


def image_data_url(image):
    png = tf.io.encode_png(tf.cast(np.round(image * 255), tf.uint8)).numpy()
    encoded = base64.b64encode(png).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def random_prediction():
    index = secrets.randbelow(len(TEST_IMAGES))
    image = TEST_IMAGES[index]
    probabilities = MODEL.predict(image[np.newaxis, ...], verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    predicted_index = int(top_indices[0])
    return {
        'index': index,
        'dataset_size': len(TEST_IMAGES),
        'actual': CLASS_NAMES[int(TEST_LABELS[index])],
        'predicted': CLASS_NAMES[predicted_index],
        'correct': predicted_index == int(TEST_LABELS[index]),
        'confidence': float(probabilities[predicted_index]),
        'top_predictions': [
            {'label': CLASS_NAMES[int(item)], 'confidence': float(probabilities[item])}
            for item in top_indices
        ],
        'image': image_data_url(image),
    }


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/':
            payload = PAGE.encode('utf-8')
            content_type = 'text/html; charset=utf-8'
        elif route == '/api/random-test-image':
            payload = json.dumps(random_prediction()).encode('utf-8')
            content_type = 'application/json; charset=utf-8'
        else:
            self.send_error(404, 'Not found')
            return

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        print(f'{self.address_string()} - {format % args}')


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8000), RequestHandler)
    print('Open http://127.0.0.1:8000 in your browser.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping server.')
        server.server_close()
