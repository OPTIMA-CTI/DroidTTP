# DroidTTP

## Overview
The rapid expansion of the Internet of Things (IoT) and mobile technologies has led to an increased reliance on Android devices for sensitive operations such as banking, online shopping, and communication. While Android remains the dominant mobile operating system, its widespread adoption has made it a prime target for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks.

Traditional malware detection methods focus primarily on binary classification, failing to provide insights into the Tactics, Techniques, and Procedures (TTPs) used by adversaries. Understanding how malware operates is essential for strengthening cybersecurity defenses.

To bridge this gap, we present **DroidTTP**, a solution designed to map Android malware behaviors to TTPs as defined by the MITRE ATT&CK framework. This system empowers security analysts with deeper insights into attacker methodologies, enabling more effective defense strategies.

## Key Contributions
- **Dataset Curation:** We curated a novel dataset explicitly designed to link MITRE ATT&CK TTPs to Android applications.
- **Problem Transformation Approach (PTA):** Utilized PTA techniques to map Android applications to both Tactics and Techniques.
- **Large Language Models (LLMs):** Implemented LLM-based approaches for fine-tuning and TTP prediction.
- **Retrieval-Augmented Generation (RAG):** Applied RAG with prompt engineering for enhanced TTP prediction.
- **Feature Selection & Data Augmentation:** Performed feature selection and data augmentation to improve classification performance.
- **Explainability:** Leveraged SHAP for interpreting model decisions.

## Repository Structure
- `dataset/` - Contains datasets for the problem transformation approach and LLM-based analysis.
- `code/` - Includes implementations for:
  - Problem Transformation Approach (PTA)
  - Feature Selection
  - Data Augmentation
  - Explainability (SHAP)
  - LLM Fine-Tuning
  - Retrieval-Augmented Generation (RAG) Approaches

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/OPTIMA-CTI/DroidTTP.git
   cd DroidTTP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the dataset and codes provided in the \dataset/' and `code/` directory.



