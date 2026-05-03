# 🏆 Diffusion LLM Leaderboard using AUP as the Metric

### 📊 Introducing a New Metric: AUP

Traditional throughput metrics (tokens per second) are hardware-dependent, making fair comparisons difficult. We introduce **AUP** (_Accuracy Under Parallelism_), a hardware-independent metric that jointly measures efficiency and performance.

<div align="center">

<img src="../asset/imgs/aup_illustration.png" alt="AUP Illustration" width="50%"/>

*AUP captures both parallelism (tokens per forward pass) and accuracy, with a weighting function that penalizes accuracy degradation*

</div>

**Key insight:** AUP uses tokens per forward (TPF) instead of tokens per second (TPS), making it device-independent. A higher AUP score means the model maintains accuracy while achieving high parallelism.


## 🏆 Diffusion LLM Leaderboard

<div align="center">

<img src="../asset/imgs/dllm_leaderboard.png" alt="dLLM Leaderboard" width="80%"/>

</div>

**We have released a dLLM Leaderboard comparing different dLLMs. You can find it at 🌐 [this blog](https://hao-ai-lab.github.io/blogs/text-diffusion/).**