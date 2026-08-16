<div align="center">

# VeriTMM

### 验证优先的多层膜光学仿真与逆向设计工作台

[![CI](https://github.com/Lihonggang-scnu/VeriTMM/actions/workflows/ci.yml/badge.svg)](https://github.com/Lihonggang-scnu/VeriTMM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0b7285.svg)](LICENSE)
[![Optical data: CC0](https://img.shields.io/badge/optical%20data-CC0-5c940d.svg)](THIRD_PARTY_NOTICES.md)

**[English](README.md) · [系统架构](docs/ARCHITECTURE.md) · [验收机制](docs/VALIDATION.md) · [示例任务](examples/tmm_tasks)**

</div>

VeriTMM 面向平面、各向同性、一维分层光学问题，把稳定正向求解、
混合相干传播、材料数据治理、可微逆向设计、光谱收敛检查、独立求解器复算
和机器可读物理证书统一在同一种可复现任务协议中。

项目的核心原则是：**优化器可以提出设计，但不能给自己颁发合格证。**
无论结果来自正向计算还是梯度优化，都必须重新进入确定性物理验收链。

![VeriTMM 计算的六周期 DBR 光谱](docs/assets/dbr_spectrum.png)

## 为什么要做这个项目

普通 TMM 示例往往在画出一条反射/透射曲线后就结束了。真正用于科研还需要
回答：用了哪一份材料数据？有没有外推？光谱是否收敛？第二套实现能否复现？
用户问题是否超出了标量 TMM 的能力边界？结果能否在另一台机器上重放？

VeriTMM 把这些问题从“使用者自己注意”变成可执行、可审计的系统合同。

## 核心能力

| 能力 | 当前实现 |
|---|---|
| 稳定正向求解 | 相干 S 矩阵主求解器、特征矩阵诊断路径、Byrnes 独立参考后端 |
| 多层膜任务 | 薄膜、DBR、一维光子晶体、缺陷腔、啁啾膜、吸收体和有限厚基底 |
| 入射条件 | 多波长、多角度、s/TE、p/TM 和非偏振光 |
| 混合相干 | 相干薄膜与光学厚层/非相干层统一计算 |
| 输出量 | R/T/A、复振幅、场分布、逐层吸收、椭偏、系统发射率、相位与色散 |
| 材料治理 | 数据集身份、适用波段、来源、插值策略和禁止静默外推 |
| 本地光学数据库 | refractiveindex.info 快照：2,712 个数据页、252,442 个 n 点、209,839 个 k 点 |
| 逆向设计 | PyTorch 批量可微 S 矩阵、多波段厚度优化、制造边界、多起点和厚度量化 |
| 物理验收 | 能力边界、无源性/能量、光谱收敛和独立求解器一致性检查 |
| 可复现性 | 不可变 JSON 任务、规范化输入、CSV 光谱、运行清单和物理证书 |

## 我们真正贡献了什么

传输矩阵法是成熟的基础物理，本项目不声称发明了 TMM 方程。我们的贡献是把
过去分散的环节组织成一条**验证优先的科研级链路**：

1. 仿真和逆向设计共享同一种不可变任务协议；
2. 能力协议会明确拒绝光栅、超表面、各向异性、非线性等超出范围的问题；
3. 材料选择成为可追溯的科研决策，数据来源、编号和波段随结果保存；
4. 可微优化得到的厚度必须由独立路径重新计算；
5. 每次合格运行生成包含收敛、能量、一致性和限制条件的物理验收证书；
6. 把“目标得分”与“物理可信”彻底分开——未达到理想指标不等于计算错误，
   高分也不能代替物理证明。

## 安装

```bash
git clone https://github.com/Lihonggang-scnu/VeriTMM.git
cd VeriTMM
python -m pip install -e .
```

需要可微优化和绘图时：

```bash
python -m pip install -e ".[optimize,plot]"
```

## 快速开始

### 运行标准 JSON 任务

```bash
python scripts/run_tmm_task.py \
  --input examples/tmm_tasks/periodic_dbr_simulation.json \
  --output-dir outputs/dbr
```

系统会写出规范化任务、完整仿真结果、CSV 光谱、运行清单，以及最重要的
`PHYSICS_ACCEPTANCE_CERTIFICATE.json`。安装 Matplotlib 后还会生成光谱图。

### 检索本地材料数据库

```bash
python scripts/search_optical_material.py TiO2 \
  --provider rii --start-nm 500 --stop-nm 800 --limit 5
```

正式复现时应明确指定 `dataset_id`。材料波段不足时默认关闭失败，不会偷偷
使用端点值外推。

### 可微逆向设计

```bash
python scripts/run_tmm_task.py \
  --input examples/tmm_tasks/antireflection_optimization.json \
  --output-dir outputs/antireflection
```

PyTorch 负责提出候选厚度；随后 NumPy/参考求解器会重新计算、校验目标并颁发
物理证书。优化模型不能修改光谱或绕过验收。

## 验收链

```mermaid
flowchart LR
    A[标准任务合同] --> B[能力边界检查]
    B --> C[材料适用范围]
    C --> D[正向求解]
    C --> O[梯度优化]
    O --> D
    D --> E[光谱收敛]
    E --> F[独立求解器复算]
    F --> G[物理验收证书]
```

程序不会通过裁剪光谱来掩盖负吸收、非有限数或能量异常。详细说明见
[验收机制](docs/VALIDATION.md)。

## 适用范围

- 相干与混合相干平面多层结构；
- 周期/啁啾多层膜、DBR、一维光子晶体和缺陷腔；
- 有损吸收膜和有限厚基底；
- 角度、偏振、逐层吸收、椭偏和场分布；
- 相位、群时延、群时延色散和 Bloch 禁带分析；
- 厚度误差蒙特卡洛与多波段可微厚度优化。

## 明确不做什么

当前是标量、各向同性、平面一维求解器，不应拿来模拟横向光栅、超表面单元、
衍射级、各向异性张量、非线性光学、散射、有限光束、三维近场、热传输或
制造化学。遇到这些问题应切换 RCWA、Berreman 4×4、FDTD 或 FEM，而不是
强行让 TMM 给出看似完整的答案。

## 测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

没有安装 PyTorch 时，可微优化专项测试会自动跳过，其余物理测试照常运行。

## 数据、引用与许可证

项目代码采用 MIT 许可证；refractiveindex.info 数据为 CC0；Byrnes `tmm`
运行依赖采用 MIT 许可证。详情及推荐引用见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。发表科研结果时，还应引用
本次运行实际选择的原始光学常数论文。

## 当前状态

`v0.1.0` 是第一个独立开源版本。物理引擎和确定性测试已经可用，但在 `v1.0`
之前，部分公共接口仍可能调整。
