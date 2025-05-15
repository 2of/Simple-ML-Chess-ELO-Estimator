# ♟️ Chess ELO Estimator

Predict player ELO ratings from raw chess move sequences using deep learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

**Live Demo**: [2of.io/chess-elo](https://2of.io/chess-elo) *(currently inactive)*

---

## 🧠 Overview

A neural network that analyzes chess games written in Standard Algebraic Notation (SAN) and predicts both players' ELO ratings.

Just supply a PGN (as a `.txt` file or similar), and run `predict.py`.  
Boom — ELO estimates, no chess sub required.

(Because Chess.com charges for that)

---

## 🔧 Key Features

- Converts SAN move sequences into `(80, 8, 8)` `int8` tensors — 8x8 board states over 80 half-moves  
- Efficient data storage using TFRecords  
- Predicts both White and Black player ratings  
- Built with **TensorFlow** + **python-chess**  
- Includes full preprocessing pipeline (lichess → JSON → TFRecords)

---

## 🤖 Models Included

You’ll find a few different models here. Some are great. Others are... vibes-based.

- **CNN Model**  
  *Does it make sense? Not really. Felt fun though*

- **Dense Model**  


- **LSTM Model**  
  Treats the game like a time series, because... well, it is I guess

> All models are trained to predict ELO only (no game outcome, no piece counts, just the vibes — quantified).

---

## 📊 Results

Todo!!

---

## 🚀 Getting Started

1. Prepare your PGN file (`.txt`, `.pgn`, etc.)
2. Run:

   ```bash
   python predict.py --input your_game_file.pgn