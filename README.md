<div align="center">

<img src="assets/logo.png" alt="Pask Logo" width="100%">


<a href='https://arxiv.org/pdf/2604.08000'><img src='https://img.shields.io/badge/arXiv-2604.08000-b31b1b.svg'></a>  
<a href='https://pask.ai'><img src='https://img.shields.io/badge/Site-Pask.ai-6D3FE1'></a>  
<a href='https://github.com/xzf-thu/Pask'><img src='https://img.shields.io/badge/GitHub-Pask-black'></a>  
<a href='https://x.com/XieZhifei14110'><img src='https://img.shields.io/badge/X-@XieZhifei14110-000000'></a>  
<a href='http://discord.gg/GQq35RaxSM'><img src='https://img.shields.io/badge/Discord-Join%20Us-5865F2'></a>  
<a href='docs/assets/images/wechat.jpg'><img src='https://img.shields.io/badge/WeChat-Join%20Group-07C160'></a>

</div>

---

## Overview

<div align="center">
  <img src="docs/assets/images/figure1.png" alt="Pask System Overview" width="100%">
  <p><i>Overview of the Pask system.</i></p>
</div>

---

## Demo Video

<div align="center">
  
[![Pask Demo Video](docs/assets/images/video-cover.jpg)](https://github.com/xzf-thu/Pask/raw/main/docs/assets/videos/demo.mp4)

*Click to watch the demo video*

</div>

---

## LatentNeeds-Bench

### Quick Start

```bash
# Clone the repo
git clone https://github.com/xzf-thu/Pask.git
cd Pask

# Setup environment
conda create -n pask python=3.10
conda activate pask
pip install -r requirements.txt
```

### Run Evaluation

```bash
# Run a model
python -m eval.run --models gpt-5-mini --level all

# Run on local vLLM
VLLM_BASE_URL=http://localhost:9000/v1 python -m eval.run --models qwen3-30b-a3b --level all

# Score with LLM-as-judge
python -m eval.score --models gpt-5-mini

# See results
python -m eval.report

# Generate LaTeX tables & figures
python -m latex.latex_fill
python -m latex.plot
```

### Results

<div align="center">
  <img src="docs/assets/images/experiments-table1.png" alt="Main Results" width="100%">
  <p><i>Table 1: Main benchmark results. IntentFlow achieves the best balanced accuracy of 84.2.</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table2.png" alt="Demand Type Results" width="100%">
  <p><i>Table 2: Results breakdown by demand type (Work, Learning, Daily).</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table3.png" alt="Multi-Turn Results" width="100%">
  <p><i>Table 3: Multi-turn accuracy degradation analysis.</i></p>
</div>

<div align="center">
  <img src="docs/assets/images/experiments-table4.png" alt="Latency Results" width="100%">
  <p><i>Table 4: Latency comparison. IntentFlow achieves 1.3-1.5s per turn.</i></p>
</div>

---

## Paper

### Abstract

**Proactivity is a core expectation for AGI.** Prior work remains largely confined to laboratory settings, leaving a clear gap in real-world proactive agent: depth, complexity, ambiguity, precision and real-time constraints. We study this setting, where useful intervention requires **inferring latent needs from ongoing context** and grounding actions in **evolving user memory** under latency and long-horizon constraints.

We first propose **DD-MM-PAS** (Demand Detection, Memory Modeling, Proactive Agent System) as a general paradigm for streaming proactive AI agent. We instantiate this paradigm in **Pask**, with streaming **IntentFlow** model for DD, a hybrid memory (workspace, user, global) for long-term MM, **PAS** infra framework and introduce how these components form a closed loop.

---

### DD-MM-PAS: A Paradigm for Proactive AI


DD-MM-PAS is a general framework for proactive AI — systems that initiate help rather than wait to be asked. It breaks proactive intelligence into three coupled functions: detecting what a user needs, remembering who the user is over time, and actually executing useful assistance.

---

### Pask-DD: IntentFlow

<div align="center">
  <img src="docs/assets/images/figure2-intentflow.png" alt="DD-MM-PAS Paradigm" width="100%">
</div>


IntentFlow is the model responsible for figuring out, in real time, whether a user needs help and what kind. As information streams in — a conversation, a lecture, a meeting — IntentFlow reads it continuously and outputs one of three decisions: stay silent, respond immediately with a quick answer, or invoke the memory system and then give deeper personalized assistance.

---

### Pask-MM: Self-Evolving Hierarchical Memory


<div align="center">
  <img src="docs/assets/images/figure3-memory.png" alt="IntentFlow Architecture" width="100%">
</div>

The memory system gives Pask a sense of who it's working with across time. It operates at three levels:

- **User Profile**: Compact profile injected directly into every interaction (fast, always-on)
- **Working Memory**: Session-level context tracking what's happening right now
- **Long-term Store**: Retrieved via search when deeper context is needed

---

### Pask-PAS: System Implementation

<div align="center">
  <img src="docs/assets/images/figure4-pas.png" alt="PAS System Architecture" width="100%">
</div>

PAS is the system backbone that turns a detected need into actual help. It connects frontend devices (glasses, phone, computer) through a server layer to a full suite of AI models and tools — web search, code execution, vision, speech recognition, and more.

---

## Citation

```bib
@article{xie2025pask,
  title={Pask: Toward Intent-Aware Proactive Agents with Long-Term Memory},
  author={Xie, Zhifei and Hu, Zongzheng and Ye, Fangda and Zhang, Xin and Chai, Haobo and Liu, Zihang and Wu, Pengcheng and Zhang, Guibin and Liao, Yue and Hu, Xiaobin and Ye, Deheng and Miao, Chunyan and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2604.08000},
  year={2025}
}
```

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<div align="center">
  <b>© 2026 Pask — Pask-Core · NTU · NUS</b>
</div>