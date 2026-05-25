# FYP-Synthetic-Conflict-Generation-Using-GANs-and-BlueSky-Simulator
# Synthetic Conflict Generation Using GANs and BlueSky Simulator

**Final Year Project**  
**Author:** Seetha Kumara Swamy  
**Supervisor:** Dr Mohamed Arif Bin Mohamed  
**Institution:** Nanyang Technological University, Singapore

---

This repository contains the codebase for the **Final Year Project: Synthetic Conflict Generation Using GANs and BlueSky Simulator**. The goal is to artificially generate realistic air traffic conflict scenarios using Generative Adversarial Networks (GANs), simulate them in the BlueSky air traffic simulator, and facilitate advanced research on conflict detection and resolution algorithms for the aviation industry.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Example Plots](#example-plots)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Conflict detection and resolution are critical components of air traffic management, ensuring safety and efficiency in the skies. This project leverages GANs, a deep learning framework, to generate synthetic but realistic air traffic conflict scenarios. These scenarios are then simulated using the [BlueSky ATM simulator](https://github.com/TUDelft-CNS-ATM/bluesky), a widely used open-source platform for air traffic management research.

**Objectives:**
- Develop a GAN model capable of generating plausible air traffic conflicts.
- Integrate the generator output with BlueSky for scenario simulation.
- Facilitate the benchmarking and advancement of conflict detection and resolution algorithms.

---

## Features

- **Synthetic Data Generation:** Utilize GANs to produce new and diverse air traffic conflict scenarios.
- **Simulation Integration:** Seamless scenario simulation in BlueSky.
- **Dataset Preparation:** Scripts for dataset preprocessing, feature extraction, and labeling.
- **Results Visualization:** Tools for visualizing generated conflicts and comparing them to real-world data.

---

## Technical Approach

### Model Architecture

- **Generator:** The generator is designed as a multilayer perceptron (MLP) that receives a noise vector and outputs features representing aircraft states and conflict attributes (e.g., position, heading, speed, altitude, time-to-conflict, etc.).
- **Discriminator:** An MLP discriminator takes real or generated scenarios and classifies them as real or fake, learning the distribution of genuine conflict data.
- **Training:** The GAN is trained using a dataset of real conflict scenarios, optimizing the adversarial loss until the generator produces scenarios indistinguishable from real samples.

### Integration with BlueSky

- Generated scenarios are formatted to match BlueSky scenario scripts.
- The integration pipeline allows batch simulation and visualization of GAN-generated conflicts inside BlueSky.

### Evaluation

- Visual similarity and statistical analysis are used to compare GAN-generated and real scenario distributions.
- Custom metrics evaluate the diversity and realism of generated conflicts.

---

## Installation

**Prerequisites:**
- Python 3.7+
- [BlueSky ATM Simulator](https://github.com/TUDelft-CNS-ATM/bluesky)
- PyTorch or TensorFlow (as required by the GAN implementation)

**Required Python Packages** (install with pip):

```bash
pip install -r requirements.txt
```

> If you haven't already, clone the BlueSky simulator and follow their installation instructions.

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/seetha-k-2004/FYP-Synthetic_Conflict_Generation_GANs_BlueSky.git
cd FYP-Synthetic_Conflict_Generation_GANs_BlueSky
```

### 2. Download or Prepare Datasets

Place your raw or processed datasets in the `data/` directory. Scripts for preprocessing and formatting are provided.

### 3. Train the GAN Model

```bash
python train_gan.py --config configs/gan_config.yaml
```

### 4. Generate New Scenarios

```bash
python generate_scenarios.py --num-scenarios 100 --output generated/
```

### 5. Simulate in BlueSky

Open BlueSky, and use the integration scripts to run generated scenarios. More instructions are in the `bluesky_integration/README.md`.

---

## Project Structure

```
FYP-Synthetic_Conflict_Generation_GANs_BlueSky/
├── bluesky_integration/
├── data/
├── models/
├── notebooks/
├── results/
├── configs/
├── requirements.txt
├── train_gan.py
├── generate_scenarios.py
└── README.md
```

- **bluesky_integration/**: Scripts to interface with BlueSky.
- **data/**: Datasets for training and testing.
- **models/**: Model architecture and training scripts.
- **notebooks/**: Experiments, EDA, and analysis.
- **results/**: Outputs, images, and evaluation metrics.
- **configs/**: Configuration files.

---

## Results

- Synthetic conflicts emulate realistic traffic patterns.
- Evaluation metrics and visualizations can be found in the `results/` folder.
- Ongoing experiments are described in `notebooks/`.

---

## Example Plots

Below are some sample visualizations generated by our pipeline, comparing real vs. synthetic scenarios:

| Scenario Comparison         | Description                               |
|----------------------------|-------------------------------------------|
| ![Real Scenario](results/real_example.png)       | Real conflict scenario from BlueSky |
| ![Synthetic Scenario](results/synthetic_example.png) | GAN-generated conflict scenario      |

- **Feature Distributions:**  
  ![Feature Histogram](results/feature_histogram.png)  
  *Histogram of key scenario features (e.g., time-to-conflict, relative velocity, etc.) illustrating similarity between real and generated datasets.*

- **Trajectory Visualizations:**  
  ![Trajectory Plot](results/trajectory_plot.png)

> For more visualizations and analysis, see the `results/` and `notebooks/` directories.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, documentation, or new features.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [BlueSky ATM Simulator](https://github.com/TUDelft-CNS-ATM/bluesky)
- [PyTorch](https://pytorch.org/) / [TensorFlow](https://www.tensorflow.org/)
- Supervisors, mentors, and collaborators

---

*For any queries, please contact [your.email@example.com].*
