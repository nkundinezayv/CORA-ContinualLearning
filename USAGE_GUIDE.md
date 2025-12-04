#!/usr/bin/env python3
"""
INFERENCE - QUICK REFERENCE GUIDE
Common usage examples for the inference script

Replace /path/to/your/image.jpg with your actual image path!
"""
Images to use for testing can be found in the repository, or you can upload your own images as long as they are within supported tasks

First run pip install -r requirements.txt

# ============================================================
# BASIC USAGE - IMAGE PREDICTIONS
# ============================================================

# MNIST Digit Recognition
python inference.py --image /path/to/your/mnist_digit.jpg --task MNIST

# CIFAR-10 Object Classification
python inference.py --image /path/to/your/cifar10_image.jpg --task CIFAR10

# Fashion MNIST (clothing items)
python inference.py --image /path/to/your/fashion_image.jpg --task FASHIONMNIST

# EMNIST Letters (alphabets)
python inference.py --image /path/to/your/letter.jpg --task EMNIST-LETTERS

# CIFAR-01 (Airplane vs Automobile)
python inference.py --image /path/to/your/vehicle.jpg --task CIFAR01

# PMNIST (Permuted MNIST)
python inference.py --image /path/to/your/pmnist.jpg --task PMNIST


# ============================================================
# TEXT PREDICTIONS
# ============================================================

# Sentiment Analysis (positive/negative)
python inference.py --text "This movie is amazing!" --task TEXT-SENT

# IMDb Reviews
python inference.py --text "Great film, highly recommend" --task TEXT-IMDB

# News Classification (World/Business/Sports/Tech)
python inference.py --text "Apple releases new iPhone model" --task TEXT-AGNEWS

# Question Answering (long/short answer)
python inference.py --text "What is machine learning?" --task TEXT-QA

# Natural Language Inference (entailment/neutral/contradiction)
python inference.py --text "A man is riding a bike on a mountain" --task TEXT-MULTINLI

# Emotion Classification
python inference.py --text "I am so happy today!" --task TEXT-GOEMOTIONS


# ============================================================
# LIST AVAILABLE TASKS
# ============================================================

# Show all 15 available tasks with number of classes
python inference.py --list-tasks


# ============================================================
# NOTE: INTERACTIVE MODE
# ============================================================

# Interactive mode (--interactive flag) is not recommended for production use
# Use explicit command-line arguments instead for reliable batch processing


# ============================================================
# AVAILABLE TASKS & CLASSES
# ============================================================

"""
IMAGE TASKS (Vision):
- MNIST: 10 classes (digits 0-9)
- PMNIST: 10 classes (permuted digits)
- FASHIONMNIST: 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot)
- EMNIST-LETTERS: 26 classes (A-Z)
- CIFAR01: 2 classes (Airplane, Automobile)
- CIFAR10: 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)

AUDIO TASKS:
- AUDIO-SPEECH: 35 classes (speech commands)
- AUDIO-ESC50: 50 classes (environmental sounds)

TEXT TASKS (NLP):
- TEXT-SENT: 2 classes (Negative, Positive) - Sentiment
- TEXT-IMDB: 2 classes (Negative, Positive) - Movie reviews
- TEXT-QA: 2 classes (Long answer, Short answer)
- TEXT-MULTINLI: 3 classes (Entailment, Neutral, Contradiction)
- TEXT-AGNEWS: 4 classes (World, Business, Sports, Technology)
- TEXT-GOEMOTIONS: 6 classes (Anger, Disgust, Fear, Joy, Sadness, Surprise)

OTHER:
- VIDEO: 10 classes
"""


# ============================================================
# OUTPUT EXAMPLE
# ============================================================

"""
$ python inference.py --image /path/to/cat.jpg --task CIFAR10

✓ Found model directory: ./models
[INIT] Using device: cpu
[LOAD] Brain loaded with 22 experts
[LOAD] 15 encoders, 15 heads

======================================================================
IMAGE PREDICTION
======================================================================

✓ Task: CIFAR10
   Prediction: Cat (3)
   Prediction Confidence: 95.23%
   Uncertainty Metric: 0.9000
"""


# ============================================================
# HOW TO USE - STEP BY STEP
# ============================================================

"""
Step 1: Identify your task
    Image tasks:
    - MNIST: handwritten digits
    - CIFAR10: objects/animals
    - FASHIONMNIST: clothing items
    - EMNIST-LETTERS: letters A-Z
    
    Text tasks:
    - TEXT-SENT: sentiment (positive/negative)
    - TEXT-GOEMOTIONS: emotions
    - TEXT-IMDB: movie reviews
    - TEXT-AGNEWS: news categories

Step 2: Get your image/text file
    For images: use .jpg, .png, etc.
    For text: just type it directly

Step 3: Replace the placeholder path
    Replace: /path/to/your/image.jpg
    With: your actual file path
    
    Examples:
    - macOS: ~/Desktop/cat.jpg
    - Linux: /home/username/images/cat.jpg
    - Windows: C:\\Users\\username\\Pictures\\cat.jpg

Step 4: Run the command
    python inference.py --image /your/actual/path.jpg --task CIFAR10

Step 5: View results
    Shows: Task name, Prediction, Confidence, Uncertainty metric
"""


# ============================================================
# PRACTICAL EXAMPLES (UPDATE PATHS)
# ============================================================

# macOS Example - CIFAR10
python inference.py --image ~/Downloads/cat.jpg --task CIFAR10

# Linux Example - MNIST
python inference.py --image /home/user/images/digit.jpg --task MNIST

# Windows Example - FASHIONMNIST
python inference.py --image C:\\Users\\YourName\\Pictures\\shirt.jpg --task FASHIONMNIST

# Text Example - Sentiment
python inference.py --text "I absolutely love this product!" --task TEXT-SENT

# Text Example - Emotion
python inference.py --text "I feel really frustrated right now" --task TEXT-GOEMOTIONS

# List all tasks
python inference.py --list-tasks