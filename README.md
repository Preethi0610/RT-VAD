# Real-Time Video Anomaly Detection (RT-VAD)

## Overview
This project implements a real time video anomaly detection system using the Flashback and ImageBind framework with live webcam input. It is part of my thesis research, where the broader goals include exploring continual learning (CL), Mixture-of-Experts architectures, and LoRA techniques to improve video anomaly detection performance and sustainability. This repository focuses on validating the Flashback and ImageBind approach on live video feeds and demonstrating its real time behavior through a Streamlit based application.

## Motivation
Video anomaly detection (VAD) identifies unusual or unexpected events in continuous video streams without requiring annotated anomaly data. Traditional systems struggle with real time performance and domain dependency. Flashback is a zero shot, real time paradigm that eliminates online language model inference by building a memory of normal and anomalous captions offline and matching incoming video segment embeddings against that memory at runtime, making it efficient and responsive.

## Features
- Real time webcam based anomaly detection using Flashback and ImageBind
- Zero shot inference without requiring labeled anomaly data
- Streamlit UI for live video demonstration
- Modular design to integrate future improvements (e.g., MoE, LoRA, continual learning)

## Architecture
1. **Video Encoder**  
   Live webcam frames are processed through ImageBind or a compatible transformer-based video encoder to extract embeddings.

2. **Memory Matching (Flashback)**  
   An offline pseudo scene memory is built from normal and anomalous text descriptions. During inference, incoming embeddings are compared against this memory to produce anomaly scores. 

3. **Streamlit Demo**  
   A simple Streamlit interface captures webcam input and displays anomaly scores and alerts in real time.

