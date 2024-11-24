# Improving Inter-Subject Transfer Learning (ISTL) in Brain-Computer Interfaces (BCI)

## Overview

This repository supports research aimed at enhancing **Inter-Subject Transfer Learning (ISTL)** for **Brain-Computer Interfaces (BCI)**. The focus is on addressing inter-subject variability in EEG data to improve model generalization across diverse subjects. Using the **EEG Motor Movement/Imagery Dataset (EEGMMIDB)** from PhysioNet, this project explores advanced clustering, feature extraction, and meta-learning techniques to minimize the need for individual fine-tuning while maintaining accuracy.

---

## Objectives

- Investigate clustering techniques for grouping individuals with similar EEG patterns.
- Evaluate the effectiveness of phenotype-based and feature-based clustering methods.
- Develop a meta-learning framework to integrate group-specific models for final predictions.
- Optimize preprocessing, feature extraction, and model training for inter-subject variability.

---

## Dataset

The research uses the [EEG Motor Movement/Imagery Dataset (EEGMMIDB)](https://www.physionet.org/content/eegmmidb/1.0.0/), which features:
- **110 subjects** performing motor imagery and movement tasks.
- Data collected via **64 EEG channels**, providing rich individual brain activity patterns.
- Applications include studying motor imagery and variability in neural signals across subjects.

---