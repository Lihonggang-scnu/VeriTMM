<div align="center">

# VeriTMM

### 面向 AI 驱动科研的可验证多层光学研究基础设施

**从 TMM 出发，把计算、优化、验证、数据与实验记录接成一条可以被 AI 持续调用的研究链路。**

[![CI](https://github.com/Lihonggang-scnu/VeriTMM/actions/workflows/ci.yml/badge.svg)](https://github.com/Lihonggang-scnu/VeriTMM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0b7285.svg)](LICENSE)

**[English](README.md) · [架构](docs/ARCHITECTURE.md) · [科研接口](docs/RESEARCH_INTERFACE.md) · [验证](docs/VALIDATION.md) · [示例](examples)**

`pip install veritmm` &nbsp;·&nbsp; [→ 快速开始](#-快速开始)

</div>

---

## VeriTMM 能做什么？

VeriTMM 具备完整的多层薄膜 TMM 基础能力：R/T/A、复振幅、相位、场分布、逐层吸收、椭偏参数、发射率、色散材料、多角度 s/p 偏振、相干/非相干混合传播、DBR 与腔体分析、PyTorch 可微 TMM、Adam/LBFGS 厚度优化、多起点搜索、灵敏度、容差/良率、稳健优化、参数扫描、实验存储、数据集生成和 Agent/JSON 接口。

作为一套 TMM 工具，它首先把该做的物理计算做好。

真正让 VeriTMM 与普通 TMM 库拉开距离的，是它把过去依赖研究者经验维持的**科学纪律**写进了程序本身。

---

# 🌟 VeriTMM 真正解决什么问题？

> 关于这背后更大的背景，见 [Instrument-Centered AI for Science](README_Instrument-Centered_AI4S.zh-CN.md)。

传统 TMM 库通常默认：**有一个懂物理的人始终在旁边。**

材料波段不够，人会发现；模型越界，人会停下来；优化结果好得离谱，人会重新检查；实验参数和材料来源，也会有人记下来。

当 AI 开始自动提出结构、批量仿真、调整优化策略、生成数据集并继续下一轮搜索时，这个人不再守在每一次计算旁边。

VeriTMM 因此把其中一部分判断变成机器可以执行的约束：

> **物理有效性是结果成立的前提——从来都是，只是过去由人来守。**

## ① Verifier-first：优化器不能给自己颁发合格证

```text
AI / Optimizer
      ↓
Design Space
      ↓
Evaluator
      ↓
Managed Physics Execution
      ↓
Preflight / Capability Gate
      ↓
TMM
      ↓
Independent Verification
      ↓
Physics Acceptance Certificate
```

优化器只负责提出候选。最终结果必须重新进入受管物理计算和独立验证。

> **高分说明它值得看，证书说明它究竟通过了什么检查。**

## ② 会算，也会拒绝

可靠的科研工具不仅要回答“结果是多少”，还应该在必要的时候回答：

> **这个问题不属于当前模型。**

VeriTMM 的 capability gate 会主动拒绝明显超出当前一维 TMM 能力范围的任务。给出一个拒绝，胜过返回一条看起来合理却站不住脚的光谱。

当前离线 AgentBench：

| 指标 | 当前结果 |
|---|---:|
| AgentBench | **85 / 85** |
| unsupported false acceptance | **0** |
| release gate | **PASS** |

“会拒绝”是一种能力。**能用可重复 benchmark 证明自己会拒绝，是另一回事。**

## ③ 不只验证数字，还验证物理关系

Physics Metamorphic Suite 检查诸如：

```text
100 nm A ≈ 50 nm A + 50 nm A
A / 0 nm B / C ≈ A / C
```

以及 lossless `R + T = 1`、passivity、正入射 s/p 等价、Fresnel 解析关系、互易性、波长网格细化一致性等。

它真正想问的是：

> **求解器有没有尊重物理结构本身应该满足的关系。**

## ④ 专门攻击 TMM 最容易失稳的地方

TMM StressBench 覆盖多层 DBR、金属和强吸收膜、超薄层、85° 掠入射、Brewster 条件等困难场景，并检查证书、能量守恒、passivity、NaN/Inf 和 solver agreement。

## ⑤ 两路 float64 求解器有争议时，还有高精度第三裁判

```text
Primary Solver ───────┐
                      │
Reference Solver ─────┼→ Locate disagreement
                      │
                      ↓
          High-Precision Referee
                 113-bit
                      ↓
      Which solver is closer?
      At which angle / pol?
      Which check is tightest?
```

可选 High-Precision Referee 使用 `mpmath` 113-bit 算术作为**高精度参考**，记录触发原因、worst-case / offending channel、两路差异和 closer solver。

它不会放宽验收标准，也不会覆写原来的 `accepted` 状态。

## ⑥ Certificate 不只说“通过”，还告诉你哪里最紧

证书的 `tightest_margin` 会指出距离失败阈值最近的检查：

```json
{
  "tightest_margin": {
    "check": "cross_solver_agreement",
    "observed_value": 2.3e-8,
    "acceptance_limit": 1e-7,
    "distance_to_limit": 7.7e-8,
    "normalized_margin": 0.77
  }
}
```

对于 energy conservation，还会记录最差位置对应的 angle、polarization 和 wavelength。

## ⑦ 材料和结果都有来历

一条 `R = 0.9984` 本身并不能告诉我们材料来自哪里、当前波长是否越界、有没有外推、哪个任务生成了结果、软件版本是什么、是否通过验证。

VeriTMM 会把材料身份、task hash、run identity、provenance、artifact hash、验证结果和 certificate 连接起来，并默认禁止静默材料外推。

## ⑧ 换算法，不必重新搭一套物理实验系统

```text
Gradient / Random / BO / CMA-ES / RL / DL / LLM Agent
                          ↓
                     Design Space
                          ↓
                      Evaluator
                          ↓
                       VeriTMM
                          ↓
                       Verifier
                          ↓
                  Experiment Record
```

算法可以变化，但 candidate identity、physics evaluation、provenance、certificate、dataset record 和 experiment lineage 保持同一种语义。

## ⑨ DatasetFactory：数据也应该能回到原始物理证据

DatasetFactory 不只生成 `input → label`。每条记录还能和 candidate、task、run、material、verification、certificate、seed、version 重新连接起来。

> **几年以后再问“这一行 label 是怎么来的？”，我们仍然希望能找到答案。**

## ⑩ 仿真工具也应该考虑 LLM 的上下文预算

科学计算很容易产生大量数组。如果每次都把原始 spectra、Monte Carlo samples、optimization history 和 dataset 直接塞进 Agent 上下文，很快就会出现：

> **上下文里全是数字，却没有空间继续思考。**

所以 VeriTMM 采用：

> **Compact by default. Detailed on demand.**

默认先返回 status、objective、constraints、physics acceptance、certificate/run identity、warning/failure 和 artifact reference；大型数组留在 artifacts 中，需要时再展开。

---

# ⚖️ 与成熟仿真软件和科研平台相比

成熟的 FDTD、FEM、RCWA 和多物理场平台，在复杂几何、全波求解、工程生态和物理覆盖上远强于当前 VeriTMM。

VeriTMM 当前聚焦的是另一个方向：

> **当 AI 开始自动使用物理求解器，怎样让计算过程本身更可验证、更可追溯、更难被误用？**

| 问题 | 传统 TMM / 成熟仿真工作流常见方式 | VeriTMM 当前设计 |
|---|---|---|
| 基础光学计算 | 成熟且覆盖广 | 完整 1D multilayer TMM |
| 复杂几何 / 全波 | 顶级平台优势明显 | 当前不覆盖 |
| 材料数据越界 | 用户配置、脚本或人工判断 | **Preflight 主动检查，默认 fail-closed** |
| 模型超出适用域 | 用户自己知道边界 | **Capability Gate 主动拒绝** |
| 优化器异常高分 | 用户决定是否接受 | **未通过物理验收不能进入可信结果链** |
| 独立数值复核 | 额外搭建工作流 | **内置 independent solver check** |
| 两路 solver 有争议 | 人工继续排查 | **可选 113-bit High-Precision Referee** |
| “通过得有多稳” | 人工读日志 | **tightest_margin + worst-case location** |
| 物理不变量验证 | 散落在测试中 | **Physics Metamorphic Suite** |
| 极端数值场景 | 取决于项目测试习惯 | **TMM StressBench** |
| 实验 provenance | 外部脚本/数据库 | **run/task/material/artifact 原生连接** |
| 换优化算法 | 容易重写 glue code | **共享 Design Space + Evaluator** |
| Agent 上下文成本 | 通常不是设计目标 | **compact response + artifact-backed detail** |
| 域外拒绝是否可测 | 很少作为独立指标 | **AgentBench + false acceptance metric** |
| 可审计性 | 商业平台内部实现通常封闭 | **开源、可检查、可修改** |

VeriTMM 当前并不打算”比顶级全波工具算得更多”。

它想做的，是另一件事：

> **一块适合 AI 自动研究长期依赖的、透明而保守的光学物理底座。**

当问题属于 TMM 的能力范围时，我们希望做到三件事：

> **算得对。**  
> **说得清。**  
> **以后还能查得回来。**

---

# 📊 当前 v0.6 候选版验证状态

当前 PR #3 候选分支已经通过全部 8 个 GitHub Actions jobs：

```text
Python 3.10       PASS
Python 3.11       PASS
Python 3.12       PASS
Torch             PASS
High Precision    PASS
AgentBench        PASS
Ruff              PASS
Build / Wheel     PASS
```

| 验收项 | 结果 |
|---|---:|
| 本地测试 | **382 passed / 1 skipped** |
| AgentBench | **85 / 85** |
| unsupported false acceptance | **0** |
| High-Precision CI gate | **PASS** |
| Python 3.10 / 3.11 / 3.12 | **PASS** |
| Real Torch gate | **PASS** |
| Ruff | **PASS** |
| Wheel / fresh install smoke | **PASS** |

测试数量是一方面，更值得看的是它们在验什么：

```text
数值正确性
+ 物理不变量
+ 域外拒绝能力
+ solver 一致性争议
+ 最差情形定位
+ 高精度参考验证
+ AI 执行契约
```

---

# 🚀 快速开始

```bash
pip install veritmm
```

```bash
veritmm describe --json
veritmm schema simulation
veritmm preflight task.json --json
veritmm run task.json --output-dir outputs/example --json
veritmm history --json
veritmm inspect RUN_ID --json
veritmm lineage RUN_ID --json
veritmm compare RUN_A RUN_B --json
veritmm benchmark --offline --json
```

可选高精度裁判：

```bash
pip install "veritmm[high_precision]"
```

---

# 从一条普通的薄膜光谱说起

做多层薄膜的人，大多写过这样的程序：

```text
定义材料
→ 设置膜层厚度
→ 扫描波长
→ 计算 R / T / A
→ 画图
```

如果全过程都有研究者盯着，很多问题可以靠经验兜住。

材料选错了，会有人发现。结果离谱了，会有人停下来。优化器突然给出一个“好得不像真的”结构，人也会多看一眼。

但如果一天运行几千个结构呢？

如果这些结构由 AI 自动提出呢？

如果生成、计算、优化、筛选、比较、数据集生成和下一轮搜索全部串起来呢？

这时，一条漂亮的光谱已经不够了。

我们开始问：材料从哪里来？当前波长是否越界？模型有没有超出能力范围？优化器找到的是好设计还是数值漏洞？两个独立求解器是否一致？如果不一致，哪一个更接近高精度参考？失败样本有没有从统计中消失？几天以后还能不能复现实验？最终“最优设计”到底经过了哪些检查？

过去这些问题常常散落在研究者的经验、脚本、笔记和记忆里。

自动化程度越来越高以后，它们需要慢慢成为程序本身的一部分。

---

# 把“可信”写进计算流程

```text
候选设计
   ↓
受管计算
   ↓
物理检查
   ↓
独立复核
   ↓
验收证书
   ↓
实验记录
```

所以一个结果除了光谱，还可以留下：

```text
run_id
task_sha256
material provenance
certificate_id
tightest_margin
worst_case_channel
artifact hash
validation result
warnings
failures
```

这些东西没有一条漂亮光谱那么显眼。

但实验做得越多，它们越重要。

---

# 可微优化：让梯度负责找路

```text
Thickness
   ↓
Differentiable TMM
   ↓
Objective
   ↓
Autograd
   ↓
Adam / LBFGS
   ↓
Candidate
   ↓
Deterministic Recompute
   ↓
Independent Validation
   ↓
Physics Certificate
```

梯度负责找路。

最终验收仍然交给确定性的物理流程。

---

# 当前物理范围

VeriTMM 当前核心仍是：

> **平面、各向同性、一维分层结构的 Transfer Matrix Method。**

适合 multilayer thin films、DBR、1D photonic crystals、Fabry–Pérot / defect cavities、absorbers、chirped multilayers、finite substrates、coherent/incoherent mixed propagation、多角度 s/p response 和支持范围内的 ellipsometry。

当前不覆盖任意二维周期光栅、metasurface 全波问题、横向衍射、一般各向异性、有限尺寸散射体、完整三维近场和非线性 Maxwell 问题。

> **对于自动化科研来说，明确拒绝一个越界问题，本身就是一种能力。**

---

# 🔭 TMM 是现在的起点

VeriTMM 当前不会为了“看起来更大”而急着把所有仿真方法塞进同一个仓库。

更重要的是先把一种研究接口做扎实：

```text
AI / Research Agent
        ↓
Problem Definition
        ↓
Design Space
        ↓
Physics Evaluation
        ↓
Verification
        ↓
Evidence
        ↓
Next Decision
```

未来如果扩展到 RCWA、FDFD、FDTD 或其他电磁方法，希望真正复用的是：

```text
Design Space
Objective
Constraint
Evaluator
Dataset
Experiment
Verifier
Evidence
```

这是愿景，不是当前能力声明。

今天的 VeriTMM 仍然从 TMM 做起。

先把这一块做深。

---

# 最后

VeriTMM 从一件很普通的事情开始：

> **算一条多层薄膜光谱。**

然后慢慢加入优化、实验记录、数据集、Agent 接口、物理不变量测试、独立复算和高精度裁判。

等这些东西真正连起来以后，我们越来越在意一个问题：

> **这些自动产生的结果，我们凭什么相信？**

我们希望 VeriTMM 最终能够同时做好两件事：

> **让计算更容易。**  
> **也让结果更值得相信。**

---

## License

MIT License

## Citation

如果 VeriTMM 对你的研究有帮助，建议在论文或实验记录中注明所使用的软件版本、材料数据集、计算设置和验证配置。
