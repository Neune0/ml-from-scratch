# Machine Learning from Scratch

A comprehensive implementation of machine learning algorithms built from the ground up, focusing on educational clarity and algorithmic understanding.

## Project Overview

This project aims to implement various machine learning algorithms using only fundamental libraries like NumPy and Matplotlib. By building these algorithms from scratch rather than using high-level libraries, we gain deeper insights into their inner workings, mathematical foundations, and optimization techniques.

## Current Status

This repository is in its early stages of development. Currently implemented:

- **Linear Regression**: Complete implementation with analytical solution and gradient descent visualization
- **KNN**: Initial structure established (in progress)

The project also includes comprehensive markdown notes on algorithm theory and a structured learning roadmap.

## Project Structure

```
ml-from-scratch/
├── bestiario.md           # Comprehensive catalog of ML algorithms
├── roadmap.md             # Learning path from fundamentals to advanced techniques
├── regressione_lineare.md # Detailed notes on linear regression theory
├── linear_regressor.py    # Implementation of linear regression
├── linear_regressor_visual.py  # Interactive visualization of gradient descent
├── knn.py                 # KNN implementation (in progress)
```

## Features

- **Pure Implementation**: Algorithms built using only NumPy, without relying on scikit-learn or other ML libraries
- **Visualization**: Visual tools to understand algorithm behavior
- **Detailed Documentation**: Each algorithm comes with theoretical explanations and mathematical foundations
- **Test/Train Splits**: Proper evaluation methodology

## Usage

Each algorithm can be run independently. For example, to see the linear regression implementation:

```bash
python linear_regressor.py
```

To visualize the gradient descent process:

```bash
python linear_regressor_visual.py
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Pillow (for saving animations)

## Future Plans

Following the roadmap in `roadmap.md`, this project will gradually implement:

1. Classification algorithms (KNN, Decision Trees, etc.)
2. Clustering algorithms (K-Means, DBSCAN, etc.)
3. Dimensionality reduction techniques (PCA, t-SNE, etc.)
4. Neural networks and deep learning components
5. Reinforcement learning algorithms

## Educational Purpose

This project is primarily for educational purposes. Each implementation focuses on clarity over optimization, making the code accessible to those learning machine learning concepts. The accompanying markdown files provide theoretical context for the implementations.

## Contributing

As this is a learning project, suggestions, corrections, and contributions are welcome! Feel free to open issues or pull requests if you spot improvements or want to add new algorithms.

## License

MIT
