# ♟️ Chess ELO Estimator

Predict player ELO ratings from raw chess move sequences using deep learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

**Live Demo**: [2of.io/chess-elo](https://2of.io/chess-elo)
(inactive currently)

## Overview

A neural network that analyzes chess moves in Standard Algebraic Notation (SAN) and predicts both players' ELO ratings.

## After being ignroed that chess.com charges to tell me my estimated elo >1 per 24hrs...... 





Key features:
- Parses SAN move strings into `(80, 8, 8)` int8 tensors (8x8 board states over 80 half-moves)
- Stores processed data efficiently as TFRecords
- Predicts ELO for both players simultaneously
- Built with TensorFlow and python-chess
- Includes full preprocessing pipeline (lichess → JSON → TFRecords)

## There's a few Different models here...


- CNN based model
    - Makes no sense but it *felt right*. 
- Dense model
- LSTM model.


Only predicts elo


### Results here: 


