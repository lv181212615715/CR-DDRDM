# CR-DDRDM: Consensus Ranking via Dual Dimensionality Reduction and Cost-Aware Optimization

A fusion framework for consensus ranking in large-scale online rating and ranking environments, integrating dual dimensionality reduction, local ranking generation, and consensus optimization.

## 📋 Project Overview

This repository implements the **CR-DDRDM (Consensus-Reaching Dual Dimensionality Reduction Decision Making)** framework for hotel ranking using large-scale online rating data. The method addresses conflicting rankings from different user segments by integrating dual dimensionality reduction with consensus optimization.

### Key Features:
- **Dual Dimensionality Reduction**: CLSC (Cosine Similarity Spectral Clustering) for object partitioning + DBSCAN for rating clustering
- **Consensus Optimization**: Particle Swarm Optimization (PSO) for minimum-cost consensus reaching
- **High Performance**: Achieves 93.75% top-10 overlap rate on Booking.com hotel data
- **Interpretable**: Transparent ranking process with clear consensus metrics

## 🏗️ Project Structure
CR-DDRDM/
│
├── clustering/ # Clustering algorithms
│ ├── CLSC.py # Cosine Similarity Spectral Clustering
│ └── DBSCAN.py # Density-based clustering with noise handling
│
├── PSO.py # Particle Swarm Optimization implementation
├── TOPSIS.py # TOPSIS multi-attribute decision making
├── Inner_feedback_1.py # Consensus feedback mechanism (Stage 1)
├── Inner_feedback_2.py # Consensus feedback mechanism (Stage 2)
├── splicing_integration.py # Cluster splicing and integration
├── init.py # Project initialization
├── test.py # Testing and validation scripts
│
├── .idea/ # IDE configuration files
├── README.md # This file
└── LICENSE # MIT License


## 📊 Methodology

### 1. Dual Dimensionality Reduction
- **Object Partitioning**: CLSC algorithm groups similar hotels using cosine similarity
- **Rating Clustering**: DBSCAN clusters user evaluations based on rating patterns

### 2. Local Ranking Generation
- TOPSIS method applied within each cluster
- Multiple ranking perspectives generated

### 3. Consensus Optimization
- Conflict detection and dynamic weighting
- PSO-based minimum-cost consensus reaching
- Global consensus score optimization

### 4. Integration
- Cluster-level sorting using custom preference functions
- Nonlinear-to-linear splicing for final global ranking

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
NumPy, SciPy, scikit-learn, pandas

