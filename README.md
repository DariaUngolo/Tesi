# 🧮 Tesi: Modular Bootstrap with Non-Abelian Discrete Symmetries

**Author:** Daria Ungolo  
**Institution:** University of Pisa – Department of Physics “E. Fermi”  
**Supervisor:** Prof. Alessandro Vichi  
**Thesis:** *Topological Anomalies and Modular Bootstrap in 2D CFTs with Discrete Symmetries*  
**Date:** 2025  

---

## 🌌 Overview

This repository contains the theoretical and numerical framework developed for my Master’s Thesis in Theoretical Physics at the University of Pisa.  
The project extends the **modular bootstrap** program to **two-dimensional conformal field theories (CFTs)** with **non-abelian discrete symmetries**, focusing on the simplest case: the permutation group \( S_3 \).

The goal is to explore how **topological defect lines (TDLs)** and **discrete anomalies** affect modular invariance, and to build a fully automated numerical pipeline capable of computing **universal bounds** on the conformal dimensions of primary operators.

This work demonstrates that the modular bootstrap — traditionally applied to theories without additional symmetries or with abelian ones — can be successfully generalized to non-abelian cases, providing a new way to explore the conformal landscape.

---

## 🧠 Main Ideas

- 🔹 Extend the **modular bootstrap** framework to include **non-abelian discrete symmetries**.  
- 🔹 Introduce **Topological Defect Lines** and analyze their fusion, modular action, and impact on the Hilbert space sectors.  
- 🔹 Derive bootstrap equations coupling partition functions across different sectors connected by \( S_3 \).  
- 🔹 Develop a **numerical pipeline** integrating:
  - Mathematica scripts for generating JSON problems,  
  - Python tools for file cleaning, scanning, and automation,  
  - SDPB for high-precision semidefinite programming optimization.  
- 🔹 Perform parametric scans over the central charge \( c \) and derivative order \( N_{	ext{max}} \).  
- 🔹 Validate the approach with the **three-state Potts model** (\( c = 0.8 \)) as a benchmark.

---

## ⚙️ Repository Files Description

| File | Description |
|------|--------------|
| **PROBLEM_to_JSON_s3_script_vm.wls** | Mathematica script that builds the modular bootstrap problem for the non-abelian \( S_3 \) symmetry. It generates the partition functions of different sectors and exports them in JSON format. |
| **PROBLEM_to_JSON_script_vm.wls** | Base Mathematica script for the modular bootstrap without additional symmetries, used for comparison and debugging. |
| **JSON_to_JSONCLEAN_OK0.py** | Python script that cleans and reformats the JSON files produced by Mathematica, removing non-numerical entries and preparing them for SDPB input. |
| **run_bisection_commentato_vm_FIXED.py** | Main Python driver that performs the bisection search for the conformal bound. It automates the entire process: JSON generation, SDPB execution, and log collection. |
| **run_bisection_S3_logger_vm1.py** | Alternative version of the bisection script with extended logging and debug information, optimized for \( S_3 \) tests. |
| **plot_Delta(c).py** | Visualization tool that reads the numerical results (e.g. `bound_results_S3_cached.json`) and plots the conformal bound \( \Delta_st(c) \) as a function of the central charge. |
| **README.md** | Project documentation — this file ✨ explaining the purpose, workflow, and usage of the entire pipeline. |  

---

## 🚀 How to Run

1. **Solve using SDPB**
   ```bash
   python3 run_bisection_##.py
   ```

2. **Visualize results**
   ```bash
   python3 plot_bounds.py
   ```

---

## 📊 Numerical Highlights

- ✅ Stable SDPB runs up to \( N_{	ext{max}} = 10 \).  
- ✅ Consistent bounds at \( c = 0.8 \) (Potts-like theory).  
- ✅ Verified modular consistency between \( \mathbb{Z}_2 \), \( \mathbb{Z}_3 \), and \( S_3 \) cases.  
- 🔄 Work in progress: higher-derivative scans and extended anomaly sectors.  

---

## 🌠 Physical Significance

This project provides a **proof of concept** that the modular bootstrap can be extended to theories with **non-abelian symmetry**.  
It establishes a bridge between **modular invariance**, **topological algebra**, and **representation theory**, showing that it is possible to treat more realistic models — where symmetry operations do not commute — within a fully non-perturbative framework.

From a physical perspective, the obtained **bounds** act as universal constraints on the conformal dimensions of primary operators, mapping the space of consistent CFTs.  
This approach opens the door to:

- 🧩 Classifying new 2D CFTs with non-abelian symmetry  
- 🌉 Connecting topological phases and their boundary conformal theories  
- ⚡ Advancing the non-perturbative study of symmetry and modularity in quantum field theory  

---

## 🔭 Future Directions

- ⚙️ Improve numerical precision and solver performance in SDPB  
- 💾 Optimize memory handling and automate large-scale scans  
- 🧮 Extend the framework to other finite groups (\( S_4 \), \( A_5 \), etc.)  
- 🪐 Explore connections with AdS\(_3\)/CFT\(_2\) dualities and topological quantum phases  

---

## 📚 Reference

If you use or adapt parts of this work, please cite:

> Daria Ungolo, *Topological Anomalies and Modular Bootstrap in 2D CFTs with Discrete Symmetries*,  
> Master’s Thesis, University of Pisa, 2025.

---

## 💫 Acknowledgements

I would like to thank **Prof. Alessandro Vichi** for his guidance and constant support,  
and the **INFN Pisa computing group** for providing the resources that made the numerical part of this project possible.  
💻 Special thanks to all the researchers and developers of **SDPB** and the **Bootstrap Collaboration** for their open-source tools and documentation.

---

##

