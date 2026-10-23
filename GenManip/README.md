# GenManip Suite

**GenManip** is a comprehensive robotics simulation suite built on **NVIDIA Isaac Sim**, designed for research in **general robotic manipulation**.
It provides an integrated platform for **data generation, benchmarking, and baseline development**, offering a unified workflow from precision scene design to large-scale dataset creation.

This repository contains installation instructions, tutorials, documentation, example benchmarks, and references for all baseline methods.

[![Paper](https://img.shields.io/badge/Paper-arXiv%20\(CVPR%202025\)-blue)](https://arxiv.org/abs/2506.10966)
[![Project Page](https://img.shields.io/badge/Website-genmanip.axi404.top-%231877F2)](https://genmanip.com/)
[![Docs](https://img.shields.io/badge/Docs-Available-brightgreen)](https://genmanip.axi404.top/overview)

---

## 📦 About GenManip

GenManip supports the full workflow—from **handcrafted scenes** to **procedurally generated large-scale datasets**.
Its streamlined toolchain allows you to easily build, customize, and share your own manipulation tasks.

The core concept is the **GenManip Package**:
Install official or community benchmarks just like adding expansion packs to a game.
Everything communicates through a black-box unified API so you can focus on model development without worrying about internal implementations.

GenManip strictly follows **LeRobot GR00t** data conventions, ensuring compatibility with modern training pipelines.

---

## 🌟 Key Highlights

* 🔌 **GenManip Package System**

  Install or publish benchmark assets with a single command — expandable like game DLCs.

* 📊 **Unified Benchmark Interface**

  Includes *GenManip Scaling Pick-and-Place*, *GenManip IROS Benchmark*, and more.
  All benchmarks share one unified communication API, making model evaluation plug-and-play.

* 🧩 **User-Friendly Docs & Config Templates**

  Rich tutorials and configuration examples help you get started in minutes.
  You can create your own benchmark or data pipeline with just a few config edits.

* 🎨 **Full-Stack Domain Randomization**

  Randomize *objects, layouts, lights, cameras, textures, rooms*, enabling robust large-scale data generation.

* 🤖 **Cross-Embodiment Support**

  Works out of the box with:

  * Franka Panda + Panda Hand
  * Franka + Robotiq 2F-85
  * Aloha Split
  * Lift2

* 📐 **Rule/Execution Set System**

  Provides a structured syntax for defining task completion logic
  (*top / left / right / front / back / in* relations + logical composition). Compute the rules and generate data by execution set, result in photorealistic manipulation data.

* 🚀 **Massive Parallel Execution**

  Run thousands of Isaac Sim instances across multiple servers.
  Stress-tested to **1500 concurrent instances** on **500× RTX 4090 (48GB)** GPUs.

* 🏭 **High-Performance Data Generation Pipeline**

  Built on **cuRobo** + generalized oracle rules.
  Scales from single GPU to hundreds of GPUs.

* 🧱 **Meta Object System**

  Flexible scene composition and object substitution for scalable dataset/benchmark creation.

---

## 🚀 Getting Started

You can launch your first benchmark or data generation pipeline in minutes.
Check out our tutorials for a step-by-step learning path — from basics to advanced usage.

👉 Full tutorials available at **[genmanip.com](https://genmanip.com)**

For questions or collaborations, feel free to open an Issue or contact:
📧 **[gaoning@pjlab.org.cn](mailto:gaoning@pjlab.org.cn)**

See our user case at: **[overview/#example-use-cases](https://genmanip.com/overview/#example-use-cases)**

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{gao2025genmanip,
  title={GenManip: LLM-driven Simulation for Generalizable Instruction-Following Manipulation},
  author={Gao, Ning and Chen, Yilun and Yang, Shuai and Chen, Xinyi and Tian, Yang and Li, Hao and Huang, Haifeng and Wang, Hanqing and Wang, Tai and Pang, Jiangmiao},
  booktitle={CVPR},
  year={2025}
}
```

Know more about our CVPR paper version at branch [archived/cvpr2025](https://github.com/InternRobotics/GenManip/tree/archived/cvpr2025)
