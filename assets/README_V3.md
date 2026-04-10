<div align="center">
<img src="assets/logo.png" alt="Pask Logo" width="100%">
</div>

> What would a truly proactive AI — one worthy of AGI — look like? 

- **Precision** → we train IntentFlow, a dedicated model for real-time latent intent classification
- **Low Latency** → we design IntentFlow with a streaming architecture for per-turn online inference
- **Long-term Modeling** → we propose a hierarchical memory module for self-evolving user modeling across sessions

In this paper, we introduce **DD-MM-PAS**, a general paradigm of Demand Detection, Memory Modeling, and Proactive Agent System, for building low-latency, online proactive AI. We instantiate each component and show how they connect into a closed loop.

<div align="center">
<a href='https://arxiv.org/pdf/2604.08000'><img src='https://img.shields.io/badge/arXiv-2604.08000-b31b1b.svg'></a>  
<a href='https://pask.ai'><img src='https://img.shields.io/badge/Site-Pask.ai-6D3FE1'></a>  
<a href='https://github.com/xzf-thu/Pask'><img src='https://img.shields.io/badge/GitHub-Pask-black'></a>  
<a href='https://x.com/XieZhifei14110'><img src='https://img.shields.io/badge/X-@XieZhifei14110-000000'></a>  
<a href='http://discord.gg/GQq35RaxSM'><img src='https://img.shields.io/badge/Discord-Join%20Us-5865F2'></a>  
<a href='docs/assets/images/wechat.jpg'><img src='https://img.shields.io/badge/WeChat-Join%20Group-07C160'></a>
</div>

<div align="center">
  <img src="docs/assets/images/figure1.png" alt="Pask System Overview" width="100%">
</div>

---

## Demo

<div align="center">
[![Pask Demo Video](docs/assets/images/video-cover.jpg)](https://github.com/xzf-thu/Pask/raw/main/docs/assets/videos/demo.mp4)
</div>


## News
- [2025.04] 🚀 Paper released: [Pask: Toward Intent-Aware Proactive Agents with Long-Term Memory](https://arxiv.org/pdf/2604.08000)
- [2025.04] 📚 LatentNeeds-Bench benchmark released
- [2025.04] 🎯 IntentFlow model weights released


## LatentNeeds-Bench

### Quick Start

```bash
git clone https://github.com/xzf-thu/Pask.git
cd Pask

conda create -n pask python=3.10
conda activate pask
pip install -r requirements.txt
```

### Evaluation

```bash
# Run a model
python -m eval.run --models gpt-5-mini --level all

# Run on local vLLM
VLLM_BASE_URL=http://localhost:9000/v1 python -m eval.run --models qwen3-30b-a3b --level all

# Score with LLM-as-judge
python -m eval.score --models gpt-5-mini

# Summarize results
python -m eval.report

# Generate LaTeX tables & figures
python -m latex.latex_fill
python -m latex.plot
```

### Results

<div align="center">
  <img src="docs/assets/images/experiments-table1.png" width="100%">
  <p><i>Table 1: Main benchmark results. IntentFlow achieves the best balanced accuracy of 84.2.</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table2.png" width="100%">
  <p><i>Table 2: Breakdown by demand type (Work, Learning, Daily).</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table3.png" width="100%">
  <p><i>Table 3: Multi-turn accuracy degradation.</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table4.png" width="100%">
  <p><i>Table 4: Latency. IntentFlow achieves 1.3–1.5 s per turn.</i></p>
</div>

---

## Paper

### Abstract

Proactivity is a core expectation for AGI. Prior work remains largely confined to laboratory settings, leaving a clear gap with real-world proactive agents: depth, complexity, ambiguity, precision, and real-time constraints. We study this setting, where useful intervention requires **inferring latent needs from ongoing context** and grounding actions in **evolving user memory** under latency and long-horizon constraints.

We propose **DD-MM-PAS** (Demand Detection, Memory Modeling, Proactive Agent System) as a general paradigm for streaming proactive AI. We instantiate this in **Pask**, with the streaming **IntentFlow** model for demand detection, a hierarchical memory system (workspace, user, global) for long-term modeling, and the **PAS** infrastructure that closes the loop from detection to action.


### DD-MM-PAS

DD-MM-PAS decomposes proactive intelligence into three coupled functions: detecting what a user needs (DD), remembering who the user is over time (MM), and executing useful assistance (PAS). Each component can be studied and improved independently, while the full system operates as a closed loop.


### Pask-DD: IntentFlow

<div align="center">
  <img src="docs/assets/images/figure2-intentflow.png" width="100%">
</div>

IntentFlow reads a live conversation continuously and outputs one of three decisions at each turn: stay silent, respond immediately, or query memory before responding. The model is trained to make this judgment under streaming constraints, with outputs calibrated to minimize both false positives (unnecessary interruptions) and false negatives (missed needs).


### Pask-MM: Hierarchical Memory

<div align="center">
  <img src="docs/assets/images/figure3-memory.png" width="100%">
</div>

The memory system operates at three levels:

- **User Profile** — compact summary injected into every inference call; always-on, near-zero latency
- **Working Memory** — session-level state tracking current task context
- **Long-term Store** — retrieved via search when session context is insufficient

The system updates continuously across sessions, allowing Pask to personalize responses based on past interactions without manual configuration.


### Pask-PAS: System Architecture

<div align="center">
  <img src="docs/assets/images/figure4-pas.png" width="100%">
</div>

PAS connects frontend devices (glasses, phone, desktop) through a server layer to the full model and tool suite — web search, code execution, vision, and speech recognition. Detected intents are routed to the appropriate tool and returned to the user within the latency budget.


## Citation

```bibtex
@article{xie2025pask,
  title={Pask: Toward Intent-Aware Proactive Agents with Long-Term Memory},
  author={Xie, Zhifei and Hu, Zongzheng and Ye, Fangda and Zhang, Xin and Chai, Haobo and Liu, Zihang and Wu, Pengcheng and Zhang, Guibin and Liao, Yue and Hu, Xiaobin and Ye, Deheng and Miao, Chunyan and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2604.08000},
  year={2025}
}
```

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

<div align="center">
  <b>© 2026 Pask — Pask-Core · NTU · NUS</b>
</div>