<div align="center">

# Leader Election in Adaptive LLM Agent Groups
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16942091.svg)](https://doi.org/10.5281/zenodo.16942091)

</div>


**Simulation and analysis of leader election dynamics in adaptive LLM agent groups (GPT-5, GPT-4, LLaMA-3).**

Code for controlled experiments on **voting behavior, network centrality, and fairness** in AI-driven multi-agent systems.

---

## 📌 Overview
This repository provides the simulation framework, experimental protocols, and analysis tools to study leadership emergence in groups of autonomous LLM agents.

### Key Features
- **Three experimental conditions**: Full election (nomination + discussion + voting), direct voting, and random baseline.
- **Metrics**: Vote distribution, centrality analysis, speaking order diversity, and initiative effects.
- **Models**: GPT-5-mini, GPT-4o-mini, LLaMA-3-70B.

---

## 🛠 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/samymih/llm-leader-election.git
   cd llm-leader-election
   ```
2. Install dependencies
3. Run simulations:
   ```bash
   python simulation.py
   ```

---

## 📂 Structure
```
.
├── writing/  # Research papers and documentation
├── results.zip # Results that I found
├── requirements.txt
└── README.md
```

---

## 📊 Results
- Vote variability across models and conditions.
- Relationship between **centrality** and electoral success.
- Impact of **initiative-taking** on leadership outcomes.

---

## 📜 Citation
If you use this work, please cite:
```
[Your citation here] © 2025 [Samy Mihoubi](https://samymihoubi.fr)
```
