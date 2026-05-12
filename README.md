# Smart Algorithm Selection System

An AI-powered web application that dynamically selects the optimal algorithmic strategy (Sorting, Dynamic Programming, etc.) based on problem characteristics and input size. 

## Features
* **AI Prediction:** Uses a Scikit-Learn Decision Tree trained on empirical benchmark data.
* **NLP Extraction:** Parses plain English problem statements to identify parameters and arrays.
* **Fair Benchmarking:** Runs algorithms in real-time using deep-copied data to prevent memory advantages.

## How to Run
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt