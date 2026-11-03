from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / 'Final_Project_Presentation_Template.pptx'
OUTPUT = ROOT / 'CIFAR10_CNN_Final_Project_Presentation.pptx'

NAVY = RGBColor(20, 38, 58)
INK = RGBColor(34, 39, 43)
MUTED = RGBColor(91, 103, 113)
ORANGE = RGBColor(205, 91, 48)
GREEN = RGBColor(38, 126, 91)
WHITE = RGBColor(255, 255, 255)
LIGHT = RGBColor(245, 247, 248)

prs = Presentation(str(TEMPLATE))
W = prs.slide_width
H = prs.slide_height


def clear_text(slide):
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False):
            shape.text_frame.clear()


def set_run(run, size=18, color=INK, bold=False, font='Aptos'):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_box(slide, x, y, w, h, text='', size=18, color=INK, bold=False,
            align=PP_ALIGN.LEFT, fill=None, line=None, margin=0.08, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = text
    for run in p.runs:
        set_run(run, size, color, bold)
    return shape


def add_rich_lines(slide, x, y, w, h, lines, fill=None, line=None, margin=0.12):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        shape.fill.solid(); shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(margin); tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin); tf.margin_bottom = Inches(margin)
    for i, item in enumerate(lines):
        text, size, color, bold = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.space_after = Pt(5)
        for run in p.runs:
            set_run(run, size, color, bold)
    return shape


def title(slide, text, kicker=None):
    if kicker:
        add_box(slide, 0.65, 0.34, 4.8, 0.25, kicker.upper(), 10, ORANGE, True)
    add_box(slide, 0.65, 0.68, 12.0, 0.62, text, 28, NAVY, True)
    add_box(slide, 0.65, 1.38, 12.0, 0.03, '', fill=ORANGE)


def footer(slide, number):
    add_box(slide, 0.65, 7.16, 8.0, 0.18, 'CIFAR-10 CNN COMPARATIVE STUDY', 8, MUTED, True)
    add_box(slide, 12.0, 7.14, 0.7, 0.2, f'{number:02d}', 9, ORANGE, True, PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=17, color=INK, fill=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        shape.fill.solid(); shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    shape.line.fill.background()
    tf = shape.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.1); tf.margin_bottom = Inches(0.08)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = '- ' + item
        p.level = 0
        p.space_after = Pt(8)
        for run in p.runs:
            set_run(run, size, color, False)
    return shape


def card(slide, x, y, w, h, heading, body, accent=ORANGE):
    add_box(slide, x, y, w, h, '', fill=LIGHT, line=RGBColor(220, 226, 230), margin=0)
    add_box(slide, x + 0.15, y + 0.13, w - 0.3, 0.3, heading, 14, accent, True)
    add_box(slide, x + 0.15, y + 0.5, w - 0.3, h - 0.62, body, 13, INK)


def picture(slide, path, x, y, w=None, h=None):
    if path.exists():
        if w and h:
            slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
        elif w:
            slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
        else:
            slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))


def table_text(slide, x, y, w, h, rows, font_size=12):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    set_run(run, font_size, WHITE if r == 0 else INK, r == 0)
    return shape


for slide in prs.slides:
    clear_text(slide)

# 1 Title
s = prs.slides[0]
add_box(s, 0.75, 0.55, 5.5, 0.28, 'DATA SCIENCE PROJECT / FINAL PRESENTATION', 11, ORANGE, True)
add_box(s, 0.75, 1.25, 11.6, 1.4, 'DEEP LEARNING-BASED\nIMAGE CLASSIFICATION ON CIFAR-10', 30, NAVY, True)
add_box(s, 0.78, 3.05, 10.3, 0.55, 'A comparative study of Baseline CNN, Mixup, and CutMix augmentation', 20, MUTED)
add_box(s, 0.8, 4.2, 5.5, 1.2, 'Student: [Your Name]\nCourse: Data Science Project\nInstitution: [Your Institution]', 15, INK)
add_box(s, 7.9, 4.45, 4.0, 0.85, 'Best test accuracy\n81.48% with Mixup', 21, WHITE, True, PP_ALIGN.CENTER, fill=ORANGE, margin=0.18, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 0.8, 6.65, 5.5, 0.3, 'October 2026', 11, MUTED, True)

# 2 Introduction
s = prs.slides[1]; title(s, 'Introduction & Problem Statement', '01 / CONTEXT')
card(s, 0.75, 1.75, 3.75, 2.0, 'Background', 'Image classification is a core computer-vision task. CIFAR-10 provides a compact benchmark for testing how CNNs learn visual patterns from small RGB images.')
card(s, 4.8, 1.75, 3.75, 2.0, 'Problem', 'CNNs can overfit training data. The project tests whether research-backed augmentation can improve generalization without changing the model architecture.')
card(s, 8.85, 1.75, 3.75, 2.0, 'Motivation', 'A fair comparison helps identify which technique gives the best balance of accuracy, loss, training cost, and practical usefulness.')
bullets(s, 0.9, 4.35, 11.7, 1.6, ['Research question: Do Mixup and CutMix improve test performance over a standard CNN?', 'Success criterion: compare all models on the same held-out CIFAR-10 test set.', 'Practical outcome: expose individual test predictions through an interactive local website.'], 18)
footer(s, 2)

# 3 Objectives
s = prs.slides[2]; title(s, 'Objectives & Scope', '02 / GOALS')
for i, (head, body) in enumerate([
    ('Build', 'Implement a baseline CNN for 10-class CIFAR-10 classification.'),
    ('Augment', 'Implement Mixup and CutMix as on-the-fly training transformations.'),
    ('Evaluate', 'Compare accuracy, loss, precision, recall, F1-score, and training time.'),
    ('Explain', 'Analyze class-level errors and expose predictions in a browser demo.'),
]):
    x = 0.8 + (i % 2) * 6.1; y = 1.85 + (i // 2) * 2.15
    card(s, x, y, 5.55, 1.65, head, body, GREEN if i == 3 else ORANGE)
add_box(s, 0.85, 6.25, 11.5, 0.45, 'Scope: one CNN architecture, 50 epochs, fixed seed 42, one alpha value per augmentation.', 16, MUTED, True, PP_ALIGN.CENTER)
footer(s, 3)

# 4 Literature
s = prs.slides[3]; title(s, 'Literature Survey & Research Gap', '03 / PRIOR WORK')
card(s, 0.8, 1.75, 3.8, 3.2, 'Mixup / Zhang et al. (2017)', 'Creates virtual samples by interpolating two images and their labels. It encourages smoother decision boundaries and acts as a regularizer.', GREEN)
card(s, 4.8, 1.75, 3.8, 3.2, 'CutMix / Yun et al. (2019)', 'Pastes a patch from one image into another and mixes labels according to the visible area. It encourages local feature reasoning.', ORANGE)
card(s, 8.8, 1.75, 3.75, 3.2, 'Research Gap', 'The techniques can behave differently on small 32x32 images and simple CNNs. This project measures their effect under a controlled local experiment.', NAVY)
add_box(s, 1.0, 5.45, 11.2, 0.7, 'Contribution: an end-to-end, reproducible comparison plus an interactive test-set prediction explorer.', 20, NAVY, True, PP_ALIGN.CENTER)
footer(s, 4)

# 5 Methodology
s = prs.slides[4]; title(s, 'Proposed System & Methodology', '04 / PIPELINE')
steps = [('1', 'Define', 'Question, scope, metrics'), ('2', 'Prepare', 'Load and normalize CIFAR-10'), ('3', 'Build', 'Shared CNN architecture'), ('4', 'Augment', 'Mixup and CutMix batches'), ('5', 'Evaluate', 'Test metrics and demo')]
for i, (n, head, body) in enumerate(steps):
    x = 0.72 + i * 2.48
    add_box(s, x, 2.0, 0.54, 0.54, n, 18, WHITE, True, PP_ALIGN.CENTER, fill=ORANGE, valign=MSO_ANCHOR.MIDDLE)
    add_box(s, x + 0.7, 1.95, 1.62, 0.35, head, 15, NAVY, True)
    add_box(s, x + 0.7, 2.35, 1.65, 0.7, body, 12, MUTED)
    if i < 4: add_box(s, x + 2.12, 2.14, 0.35, 0.05, '', fill=ORANGE)
card(s, 0.85, 4.25, 5.6, 1.35, 'Key novelty', 'The comparison keeps the architecture and core training conditions consistent, isolating the effect of augmentation.', GREEN)
card(s, 6.85, 4.25, 5.55, 1.35, 'Outputs', 'Saved .h5 models, JSON metrics, training plots, confusion matrices, and a local prediction website.', NAVY)
footer(s, 5)

# 6 Architecture
s = prs.slides[5]; title(s, 'System Architecture', '05 / MODEL')
add_box(s, 0.95, 1.8, 2.1, 0.62, '32x32x3\nRGB image', 17, WHITE, True, PP_ALIGN.CENTER, fill=NAVY, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 3.45, 1.8, 2.35, 0.62, 'Conv blocks\n32 -> 64 -> 128', 16, WHITE, True, PP_ALIGN.CENTER, fill=ORANGE, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 6.25, 1.8, 2.0, 0.62, 'Flatten +\nDense 512', 16, WHITE, True, PP_ALIGN.CENTER, fill=GREEN, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 8.75, 1.8, 2.35, 0.62, 'Softmax\n10 classes', 16, WHITE, True, PP_ALIGN.CENTER, fill=NAVY, valign=MSO_ANCHOR.MIDDLE)
for x in [3.02, 5.83, 8.28]: add_box(s, x, 2.08, 0.35, 0.05, '', fill=ORANGE)
card(s, 0.85, 3.05, 3.7, 2.1, 'Regularization', 'Dropout 0.25 after each convolutional block and Dropout 0.5 before the output layer.', ORANGE)
card(s, 4.8, 3.05, 3.7, 2.1, 'Training', 'Adam optimizer, categorical cross-entropy, batch size 64, 50 epochs, 1,341,226 parameters.', GREEN)
card(s, 8.75, 3.05, 3.7, 2.1, 'Output', 'A probability distribution over airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.', NAVY)
footer(s, 6)

# 7 Requirements
s = prs.slides[6]; title(s, 'Hardware & Software Requirements', '06 / ENVIRONMENT')
card(s, 0.85, 1.8, 3.8, 3.8, 'Hardware', 'CPU training environment\nIntel Core i9-14900HX\n32 GB RAM\nNVIDIA RTX 4070 available but not used for TensorFlow training on Windows\n\nTraining time: about 16-17 minutes per model', ORANGE)
card(s, 4.85, 1.8, 3.8, 3.8, 'Software', 'Windows environment\nPython 3.14 environment\nTensorFlow 2.22.0-rc0\nKeras API\nNumPy 2.5.4\nscikit-learn 1.9.1', GREEN)
card(s, 8.85, 1.8, 3.7, 3.8, 'Tools & Deployment', 'Matplotlib and Seaborn for plots\nJupyter for exploration\nSaved H5 models and JSON metrics\nLocal Python web server at 127.0.0.1:8000', NAVY)
footer(s, 7)

# 8 Dataset
s = prs.slides[7]; title(s, 'Dataset & Data Understanding', '07 / DATA')
for i, (value, label) in enumerate([('60,000', 'RGB images'), ('50,000', 'training images'), ('10,000', 'test images'), ('10', 'target classes')]):
    x = 0.85 + i * 3.02
    add_box(s, x, 1.8, 2.55, 1.0, value, 27, ORANGE, True, PP_ALIGN.CENTER, fill=LIGHT, valign=MSO_ANCHOR.MIDDLE)
    add_box(s, x, 2.82, 2.55, 0.38, label, 13, MUTED, True, PP_ALIGN.CENTER)
card(s, 0.85, 3.65, 5.55, 1.8, 'Classes', 'Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Each test class contributes 1,000 images.', GREEN)
card(s, 6.85, 3.65, 5.55, 1.8, 'Data quality', 'Balanced class distribution, no missing values, 32x32 pixel RGB format, and normalized values in the [0, 1] range.', NAVY)
add_box(s, 0.9, 6.05, 11.5, 0.45, 'The test set remained untouched during training and was used for final comparison.', 16, ORANGE, True, PP_ALIGN.CENTER)
footer(s, 8)

# 9 Preprocessing
s = prs.slides[8]; title(s, 'Data Preprocessing & Exploratory Analysis', '08 / PREPROCESSING')
picture(s, ROOT / 'results' / 'mixup_cnn_training_history.png', 0.75, 1.75, w=6.0)
card(s, 7.15, 1.8, 5.2, 1.35, 'Loading', 'CIFAR-10 batch files are read from the extracted dataset and reshaped from channel-first storage to 32x32x3 images.', ORANGE)
card(s, 7.15, 3.35, 5.2, 1.35, 'Normalization', 'Each pixel is divided by 255 so model inputs lie between 0 and 1.', GREEN)
card(s, 7.15, 4.9, 5.2, 1.35, 'Labels', 'Integer class labels are converted to one-hot vectors for categorical cross-entropy.', NAVY)
footer(s, 9)

# 10 Modules
s = prs.slides[9]; title(s, 'Module-wise Description', '09 / IMPLEMENTATION')
modules = [
    ('Data utilities', 'Load, normalize, encode labels, and create the shared CNN.'),
    ('Baseline CNN', 'Train the reference model without synthetic augmentation.'),
    ('Mixup CNN', 'Mix image and label pairs on every training batch.'),
    ('CutMix CNN', 'Replace image patches and adjust labels by area.'),
    ('Evaluation + web app', 'Save metrics, plots, models, and serve random test predictions.')]
for i, (head, body) in enumerate(modules):
    x = 0.8 + (i % 3) * 4.05; y = 1.8 + (i // 3) * 2.25
    card(s, x, y, 3.65, 1.65, head, body, [ORANGE, GREEN, NAVY, ORANGE, GREEN][i])
footer(s, 10)

# 11 Algorithms
s = prs.slides[10]; title(s, 'Algorithms & Technologies Used', '10 / TECHNIQUES')
card(s, 0.8, 1.8, 3.75, 3.45, 'CNN classifier', 'Chosen because convolutions learn local edges, textures, and object patterns directly from images.\n\nKey setting: 3 convolutional blocks with increasing filter counts.', NAVY)
card(s, 4.8, 1.8, 3.75, 3.45, 'Mixup', 'Chosen to regularize the model through smooth interpolation.\n\nKey setting: alpha = 0.2, applied per training batch.', GREEN)
card(s, 8.8, 1.8, 3.75, 3.45, 'CutMix', 'Chosen to test localized feature learning by hiding and replacing image regions.\n\nKey setting: alpha = 1.0, area-adjusted labels.', ORANGE)
add_box(s, 1.0, 5.8, 11.2, 0.55, 'Selection criterion: maximize held-out test accuracy while keeping the architecture and evaluation protocol consistent.', 16, NAVY, True, PP_ALIGN.CENTER)
footer(s, 11)

# 12 Implementation
s = prs.slides[11]; title(s, 'Implementation & Model Building', '11 / TRAINING')
card(s, 0.8, 1.8, 3.7, 3.6, 'Pipeline', '1. Load 50,000 training and 10,000 test images\n2. Normalize pixels\n3. One-hot encode labels\n4. Shuffle training batches\n5. Apply augmentation\n6. Train with Adam\n7. Evaluate on held-out test set', ORANGE)
card(s, 4.8, 1.8, 3.7, 3.6, 'Mixup equation', 'x_mix = lambda*x_i + (1-lambda)*x_j\n\ny_mix = lambda*y_i + (1-lambda)*y_j\n\nlambda sampled from Beta(alpha, alpha), alpha = 0.2', GREEN)
card(s, 8.8, 1.8, 3.7, 3.6, 'CutMix equation', 'A patch from image j replaces part of image i.\n\ny_mix = lambda*y_i + (1-lambda)*y_j\n\nlambda is corrected using the actual retained image area.', NAVY)
footer(s, 12)

# 13 Prototype
s = prs.slides[12]; title(s, 'Working Prototype & Output Screens', '12 / DEMO')
picture(s, ROOT / 'results' / 'mixup_cnn_confusion_matrix.png', 0.85, 1.8, w=5.0)
card(s, 6.2, 1.8, 2.8, 3.7, 'Interactive explorer', 'The local website randomly selects one of 10,000 test images and displays:\n\n- true label\n- predicted label\n- correct / incorrect status\n- confidence\n- top three predictions', GREEN)
card(s, 9.25, 1.8, 3.1, 3.7, 'How to launch', 'python web_app.py\n\nOpen:\nhttp://127.0.0.1:8000\n\nImages appear pixelated when enlarged because CIFAR-10 images are only 32x32 pixels.', ORANGE)
footer(s, 13)

# 14 Results
s = prs.slides[13]; title(s, 'Results & Performance Comparison', '13 / RESULTS')
rows = [
    ['Model', 'Test acc.', 'Test loss', 'Time', 'Val acc.'],
    ['Baseline', '79.24%', '0.6993', '1015 s', '81.10%'],
    ['Mixup', '81.48%', '0.5780', '963 s', '92.78%'],
    ['CutMix', '80.30%', '0.5952', '1031 s', '89.20%'],
]
table_text(s, 0.8, 1.8, 6.3, 2.4, rows, 13)
picture(s, ROOT / 'results' / 'mixup_cnn_training_history.png', 7.35, 1.75, w=5.25)
add_box(s, 0.95, 4.75, 11.4, 0.8, 'Best result: Mixup reached 81.48% test accuracy and reduced test loss to 0.5780.', 22, WHITE, True, PP_ALIGN.CENTER, fill=GREEN, margin=0.18, valign=MSO_ANCHOR.MIDDLE)
footer(s, 14)

# 15 Analysis
s = prs.slides[14]; title(s, 'Analysis & Key Findings', '14 / INTERPRETATION')
picture(s, ROOT / 'results' / 'mixup_cnn_confusion_matrix.png', 0.75, 1.75, w=5.25)
card(s, 6.25, 1.8, 2.95, 1.45, 'Mixup wins', '+2.24 percentage points over baseline. Lowest test loss and highest validation accuracy.', GREEN)
card(s, 9.45, 1.8, 2.95, 1.45, 'CutMix helps', '+1.06 percentage points over baseline, but trails Mixup.', ORANGE)
card(s, 6.25, 3.55, 6.15, 1.45, 'Hard classes', 'Cats, dogs, and birds remain challenging because their visual patterns overlap at 32x32 resolution.', NAVY)
card(s, 6.25, 5.25, 6.15, 1.15, 'Strong classes', 'Automobiles, ships, trucks, and frogs produced the strongest F1-scores across models.', GREEN)
footer(s, 15)

# 16 Testing
s = prs.slides[15]; title(s, 'Testing & Validation', '15 / VERIFICATION')
card(s, 0.8, 1.8, 3.75, 3.3, 'Installation test', 'test_installation.py verified TensorFlow, NumPy, Matplotlib, scikit-learn, Seaborn, and all local modules.\n\nResult: 3/3 tests passed.', GREEN)
card(s, 4.8, 1.8, 3.75, 3.3, 'Model evaluation', 'evaluate_mixup.py loaded the saved model and evaluated all 10,000 test images.\n\nObserved Mixup accuracy: 81.48%.', ORANGE)
card(s, 8.8, 1.8, 3.75, 3.3, 'Interactive validation', 'web_app.py loads the same saved model, selects a random test sample, encodes it for the browser, and returns prediction details.', NAVY)
add_box(s, 1.0, 5.75, 11.2, 0.5, 'Validation principle: use real held-out data and real model outputs, not mocked predictions.', 16, NAVY, True, PP_ALIGN.CENTER)
footer(s, 16)

# 17 Challenges
s = prs.slides[16]; title(s, 'Challenges & Limitations', '16 / REFLECTION')
card(s, 0.8, 1.8, 3.75, 3.55, 'Challenges faced', 'Small 32x32 images make fine-grained recognition difficult.\n\nWindows TensorFlow did not use the available NVIDIA GPU.\n\nA hardcoded path in the installation test initially broke local imports.', ORANGE)
card(s, 4.8, 1.8, 3.75, 3.55, 'How addressed', 'Normalized data consistently.\n\nKept architecture and core settings aligned.\n\nReplaced hardcoded paths with workspace-relative paths.\n\nAdded a direct evaluation script and web demo.', GREEN)
card(s, 8.8, 1.8, 3.75, 3.55, 'Limitations', 'Only one alpha value per method.\n\nNo early stopping or learning-rate schedule.\n\nSimple CNN without batch normalization or residual blocks.\n\nOne main seed and training configuration.', NAVY)
footer(s, 17)

# 18 Conclusion
s = prs.slides[17]; title(s, 'Conclusion', '17 / OUTCOME')
add_box(s, 0.85, 1.8, 3.0, 1.2, '81.48%', 34, WHITE, True, PP_ALIGN.CENTER, fill=GREEN, margin=0.18, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 0.85, 3.1, 3.0, 0.45, 'Best test accuracy: Mixup', 15, GREEN, True, PP_ALIGN.CENTER)
bullets(s, 4.35, 1.85, 7.9, 3.4, ['The baseline CNN achieved 79.24% test accuracy.', 'Mixup achieved the best result at 81.48%, improving the baseline by 2.24 points.', 'CutMix achieved 80.30%, improving the baseline by 1.06 points.', 'The project met its goals: implementation, fair comparison, evaluation, and an interactive prediction demo.', 'Augmentation improved generalization without changing the CNN architecture.'], 19)
add_box(s, 1.0, 5.85, 11.2, 0.55, 'Main takeaway: Mixup was the strongest method in this experiment.', 20, NAVY, True, PP_ALIGN.CENTER)
footer(s, 18)

# 19 Future scope
s = prs.slides[18]; title(s, 'Future Scope & Applications', '18 / NEXT STEPS')
card(s, 0.8, 1.8, 3.75, 3.5, 'Improve performance', 'Tune alpha values, add learning-rate schedules, batch normalization, early stopping, and stronger CNN backbones such as ResNet or WideResNet.', GREEN)
card(s, 4.8, 1.8, 3.75, 3.5, 'Expand analysis', 'Run multiple seeds, report mean and standard deviation, evaluate CIFAR-10-C corruptions, and add saliency or Grad-CAM explanations.', ORANGE)
card(s, 8.8, 1.8, 3.75, 3.5, 'Applications', 'Educational model inspection, classroom demonstrations, augmentation research, and a foundation for a larger computer-vision dashboard or API.', NAVY)
footer(s, 19)

# 20 Team
s = prs.slides[19]; title(s, 'Team Contribution', '19 / CONTRIBUTIONS')
add_box(s, 1.0, 1.95, 11.2, 0.6, 'Replace the placeholders below with your actual team members and registration numbers.', 18, MUTED, True, PP_ALIGN.CENTER)
table_text(s, 1.1, 3.0, 11.0, 2.1, [
    ['Member', 'Contribution'],
    ['[Name / Reg. No.]', 'Dataset preparation, baseline CNN, documentation'],
    ['[Name / Reg. No.]', 'Mixup and CutMix implementation, training'],
    ['[Name / Reg. No.]', 'Evaluation, visualizations, and interactive website'],
], 15)
footer(s, 20)

# 21 References
s = prs.slides[20]; title(s, 'References & Project Links', '20 / SOURCES')
refs = [
    '1. Zhang et al. (2017), “mixup: Beyond Empirical Risk Minimization,” arXiv:1710.09412.',
    '2. Yun et al. (2019), “CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,” ICCV.',
    '3. Krizhevsky & Hinton (2009), “Learning Multiple Layers of Features from Tiny Images.”',
    '4. Abadi et al. (2016), “TensorFlow: A System for Large-Scale Machine Learning.”',
]
bullets(s, 0.9, 1.75, 11.5, 2.2, refs, 16)
card(s, 0.9, 4.45, 5.55, 1.2, 'Dataset', 'CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html', GREEN)
card(s, 6.85, 4.45, 5.55, 1.2, 'Project artifacts', 'Source code, saved models, metrics, plots, and web demo are included in the project workspace.', ORANGE)
footer(s, 21)

# 22 Closing
s = prs.slides[21]
add_box(s, 0.9, 1.0, 11.5, 0.45, 'DEEP LEARNING-BASED IMAGE CLASSIFICATION ON CIFAR-10', 12, ORANGE, True, PP_ALIGN.CENTER)
add_box(s, 1.0, 2.0, 11.3, 0.9, 'THANK YOU', 40, NAVY, True, PP_ALIGN.CENTER)
add_box(s, 1.0, 3.2, 11.3, 0.55, 'Questions & Discussion', 23, GREEN, True, PP_ALIGN.CENTER)
add_box(s, 2.0, 4.45, 9.3, 0.85, 'Baseline: 79.24%    |    Mixup: 81.48%    |    CutMix: 80.30%', 18, WHITE, True, PP_ALIGN.CENTER, fill=NAVY, margin=0.18, valign=MSO_ANCHOR.MIDDLE)
add_box(s, 1.0, 6.3, 11.3, 0.35, 'Student: [Your Name]  |  Guide: [Guide Name]  |  [Institution]', 12, MUTED, False, PP_ALIGN.CENTER)

prs.save(str(OUTPUT))
print(f'Created {OUTPUT}')
