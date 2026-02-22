# Graduation Project

**Title:** Communication–Perception Collaborative Navigation and Trajectory Planning Algorithms for Multi-Robot Systems

---

## 1 Project Description

### 1.1 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 1.2 Algorithms

The project implements two reinforcement learning (RL) algorithms for navigation in a communication-aware multi-robot setting.

- **DQN (Deep Q-Network)**  
  Single-agent navigation. The agent learns a policy via Q-learning with a neural network that takes **state** and **target** as input: \(Q(s, a \mid \text{target})\). The network uses state embedding, target embedding, and relative position (distance/direction to target). Training uses experience replay, a target Q-network, and \(\varepsilon\)-greedy exploration. The reward can incorporate communication quality (e.g., BER from the NOMA/SIC model).

- **MADQN (Multi-Agent DQN)**  
  Multi-robot navigation where each robot has its own Q-network and replay buffer. All agents act in parallel; each agent moves toward its own target. The same TD learning and target-network update are applied per agent. The environment reward can include SIC-based BER so that trajectories are optimized for both reaching goals and maintaining link quality.

### 1.3 How to Run

Run from the project root.

#### Training

```bash
# Train DQN (single-agent)
python main.py --model dqn --mode train

# Train MADQN (multi-agent)
python main.py --model madqn --mode train
```

**Training outputs:**

| Output | Location |
|--------|----------|
| Model checkpoints | `models/dqn_model.pth`, `models/madqn_model.pth` |
| Training curves (plots) | `rl_algorithms/plot/figs/` (e.g. `dqn_return.png`, `madqn_return.png`, `madqn_ber.png`). You can copy these to `results/Train/` for display. |
| Logs (MADQN training) | `logs/` (e.g. `madqn_YYYYMMDD_HHMM.log`) |

#### Testing

```bash
# Test DQN (loads models/dqn_model.pth by default)
python main.py --model dqn --mode test [--model_path PATH] [--max_steps N] [--quiet]

# Test MADQN (loads models/madqn_model.pth by default)
python main.py --model madqn --mode test [--model_path PATH] [--max_steps N] [--quiet]
```

For trajectory visualization (GIF and last-frame PNG), run the test scripts after training:

```bash
# Single-agent: trajectory and snapshot saved under results/
python -m rl_algorithms.test.test_dqn

# Multi-agent: trajectory and snapshot saved under results/
python -m rl_algorithms.test.test_madqn
```

**Test outputs:**

| Output | Location |
|--------|----------|
| GIF animations | `results/gif/` (e.g. `dqn_pretrained_test.gif`, `madqn_pretrained_test04.gif`) |
| Last-frame snapshots | `results/png/` (e.g. `dqn_pretrained_test_last_frame.png`, `madqn_pretrained_test04_last_frame.png`) |

### 1.4 Training Results

Training curves and metrics are stored in `results/Train/`.

**DQN (Single-Agent)**

| DQN Training Curve |
|--------------------|
| <img src="results/Train/dqn/dqn_return.png" width="320" alt="DQN training curve" /> |

**MADQN (4-Agent)**

| MADQN 4-Agent Return | MADQN 4-Agent BER |
|----------------------|-------------------|
| <img src="results/Train/madqn_4/madqn_return.png" width="320" alt="MADQN return" /> | <img src="results/Train/madqn_4/madqn_ber.png" width="320" alt="MADQN BER" /> |

*DQN training curve; MADQN 4-agent episodic return and BER (Bit Error Rate) during training.*

### 1.5 Running Results

Trajectories and last-frame snapshots are stored in `results/`.

#### DQN (Single-Agent)

| Trajectory Animation | Last Frame |
|----------------------|------------|
| <img src="results/gif/dqn_pretrained_test.gif" width="360" alt="DQN trajectory" /> | <img src="results/png/dqn_pretrained_test_last_frame.png" width="360" alt="DQN last frame" /> |

*Single-agent DQN navigation trajectory and terminal snapshot.*

#### MADQN (Multi-Agent, 4 Agents)

| 1 Agent (test01) | 2 Agents (test02) | 4 Agents (test04) |
|------------------|-------------------|-------------------|
| ![MADQN test01](results/gif/madqn_pretrained_test01.gif) | ![MADQN test02](results/gif/madqn_pretrained_test02.gif) | ![MADQN test04](results/gif/madqn_pretrained_test04.gif) |

*Multi-agent MADQN trajectory animations with different numbers of agents.*

| MADQN Last Frame (test01) | MADQN Last Frame (test02) | MADQN Last Frame (test04) |
|--------------------------|---------------------------|---------------------------|
| ![MADQN test01 last frame](results/png/madqn_pretrained_test01_last_frame.png) | ![MADQN test02 last frame](results/png/madqn_pretrained_test02_last_frame.png) | ![MADQN test04 last frame](results/png/madqn_pretrained_test04_last_frame.png) |

#### Other Result Files

| Result | Description |
|--------|-------------|
| `results/compare/madqn_pretrained_test02.gif`, `madqn_pretrained_test02_minus.gif` | MADQN comparison experiments. |
| `results/random.gif` | Random policy baseline. |

---

## 2 System Modeling

### 2.1 System Description

<img src="figs/2-1.system_model.png" alt="System Model" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.9em; color: #555;">Fig. 1-1: System model</p>

Indoor system: one access point with **$N$ antennas**, **$K$ single-antenna robots** ($k \in \mathcal{K} = \{1,2,\dots,K\}$), and obstacles. Time step $t \in \{1,2,\dots,T_k\}$. Access point position **$q_{AP}=(x_{AP},y_{AP},z_{AP})$**; robot $k$ at $t$: **$q_{k}=(x_{k},y_{k},z_{k})$**.

### 2.2 Channel Model

Indoor SL (Sparse clutter Low BS) scenario; downlink channel coefficient (path loss + Rayleigh small-scale fading):

$$h_k(t) = PL_k(t) - 10\log_{10}(g_k(t))$$

Path loss uses ABG in SL: $f_c = 3.5\,\text{GHz}$, $d$ = AP–robot distance:

$$PL_{LOS}(f_c, d) = 31.84 + 21.50\log_{10}(d) + 19.00\log_{10}(f_c) \quad \text{(Eq. 2-2)}$$

$$PL_{SL}(f_c, d) = 33 + 25.50\log_{10}(d) + 20\log_{10}(f_c) \quad \text{(Eq. 2-3)}$$

$$PL_{NLOS} = \max(PL_{SL}, PL_{LOS}) \quad \text{(Eq. 2-4)}$$

Channel vector (array response + gain): $\beta_k(t) = 10^{-h_k(t)/20}$; $D = \lambda/2$:

$$\tilde{h}_k(t) = \beta_k(t)(\alpha_k^{\text{transmit}})^H, \quad \alpha_k^{\text{transmit}} = \frac{1}{\sqrt{N_t}} \left[1, e^{-j\pi\sin\theta}, \ldots, e^{-j\pi(N_t-1)\sin\theta}\right]^T \quad \text{(Eq. 2-5, 2-6)}$$

### 2.3 Communication Model

#### 2.3.1 NOMA Grouping

Robots sorted by $|h_k|^2$ (descending); pair rank $m$ with rank $K/2+m$ into $M = K/2$ clusters ($K$ even).

#### 2.3.2 Diagonalization Precoding

For cluster $m$, precoder $\mathbf{w}_m$ lies in the null space of channels of other clusters: $\tilde{\mathbf{H}}_{m,i} \mathbf{w}_m = \mathbf{0}$ (Eq. 2-7). SVD of $\tilde{\mathbf{H}}_{m,i}$ yields $\tilde{\mathbf{V}}_m^{(0)}$ (null space); SVD of $\mathbf{H}_m \tilde{\mathbf{V}}_m^{(0)}$ yields $\mathbf{V}_m^{(1)}$. Then:

$$\mathbf{w}_m = \tilde{\mathbf{V}}_m^{(0)} \mathbf{V}_m^{(1)} \quad \text{(Eq. 2-8--2-10)}$$

#### 2.3.3 SIC and URLLC Error Rate

Received signal (cluster $m$, two users):

$$\begin{bmatrix} \mathbf{y}_{m,1} \\ \mathbf{y}_{m,2} \end{bmatrix} = \begin{bmatrix} \mathbf{h}_{m,1} \\ \mathbf{h}_{m,2} \end{bmatrix} \mathbf{w}_m \mathbf{s}_m + \begin{bmatrix} \mathbf{n}_{m,1} \\ \mathbf{n}_{m,2} \end{bmatrix} \quad \text{(Eq. 2-11)}$$

Power allocation: weak user gets more power; SIC constraint $(P_{m,2}-P_{m,1})\beta_{m,1} \geq \rho_{\min}$. After SIC:

$$SINR_{m,1} = \frac{P_{m,1} |\mathbf{h}_{m,1} \mathbf{w}_m|^2}{\sigma^2}, \quad SINR_{m,2} = \frac{P_{m,2}|\mathbf{h}_{m,2}\mathbf{w}_m|^2}{P_{m,1}|\mathbf{h}_{m,2}\mathbf{w}_m|^2 + \sigma^2} \quad \text{(Eq. 2-12, 2-13)}$$

Finite-blocklength decoding error (URLLC): $Q(\xi) = \frac{1}{\sqrt{2\pi}}\int_{\xi}^{\infty} e^{-t^2/2}\,dt$, $V = 1-(1+SINR_{m,i})^{-2}$:

$$\epsilon_{m,i}(t) = Q\left(\ln 2 \sqrt{\frac{N}{V}}\left(\log_2(1+SINR_{m,i})-\frac{D}{N}\right)\right) \quad \text{(Eq. 2-14–2-16)}$$

$N$ = block length, $D$ = packet size. Analysis focuses on the weaker user's error rate.
