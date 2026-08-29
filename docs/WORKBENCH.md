# TMM“瑞士军刀”工作台：当前主线与使用边界

## 一句话说明

这次改造把旧版“输入几层材料，只算一条反射光谱”的小工具，升级为一条可复用的物理闭环：

```text
标准任务协议
  → 明确选择材料数据及适用波段
  → 正向计算
  → 物理审计
  → 可微厚度优化
  → 独立求解器复核
  → 光谱、材料来源、运行清单与验收报告
```

自然语言理解暂不属于本阶段。以后只需增加一个语言模型，把用户描述填写到固定协议中；物理层不依赖模型记忆，也不允许模型绕过单位、材料范围或验收规则。

## 统一输入协议

核心对象位于 `tmm_engine/schemas.py`：

- `MediumSpec`：入射介质或出射介质；
- `LayerSpec`：有限厚度层，可使用材料数据库或论文报告的常数折射率；
- `StackSpec`：从入射侧到出射侧的完整层序；
- `SpectralGrid`：波长范围或明确的波长数组；
- `IlluminationSpec`：入射角与偏振；
- `SimulationTask`：正向仿真任务；
- `SpectralTarget`：优化目标，可表达“匹配”“至少达到”或“至多不超过”，并支持平均性能和最差点性能；
- `OptimizationTask`：仿真、目标和优化参数的组合。

公共协议中，波长和厚度统一使用 nm。材料数据库内部使用 µm，换算只在材料接口边界发生一次，避免重复换算。

## 材料接口

`MaterialRegistry`（材料注册器）统一管理两类来源：

1. 14 份项目内置 CSV：速度快、版本稳定，默认优先；
2. `rii_cache.db`：refractiveindex.info 的本地 SQLite 镜像，当前含 2712 个数据页、252442 个折射率采样点和 209839 个消光系数采样点。

材料选择会记录数据源、数据集编号、原始页面、文件路径和有效波段。超出有效波段时默认报错；只有用户明确允许时才进行端点外推，并在结果中留下警告。

使用示例：

```powershell
python scripts/search_optical_material.py TiO2 --provider rii `
  --start-nm 500 --stop-nm 800 --dataset-id 418 `
  --export-csv outputs/tio2_418.csv
```

这里采用本地镜像而不是每次访问网页，优点是速度快、可复现、不会受网页结构和网络影响。后续可以更新镜像版本，但不能在同一次实验中悄悄切换数据集。

## 正向仿真能力

默认使用稳定的散射矩阵求解器。另有两个后端：

- 特征矩阵：用于简单结构的交叉诊断，厚吸收层可能数值不稳定；
- Byrnes `tmm`：独立维护的参考实现，负责相干/非相干混合、有限厚基底、层内场、逐层吸收和椭偏参数。

当前可完成：

- 多层增透膜、高反膜、吸收膜；
- 一维光子晶体、缺陷腔和啁啾堆栈；
- 多波段、多角度、s/p/非偏振的 R/T/A；
- 显式有限基底和厚基底非相干传播；
- 层内电场与吸收分布；
- 椭偏参数；
- Bloch 禁带判断；
- 厚度误差蒙特卡洛分析；
- 半球平均光谱；
- 两种不同的热辐射语义：有限层吸收率和“包含吸收基底的整个系统发射率”。

旧接口不能描述有限基底，因为它没有基底厚度和背面介质。现在旧接口遇到这种请求会明确拒绝，不再返回一个看似正常但实际仍是半无限基底的结果。

## SpecFormer 可微优化如何接入

保留了 SpecFormer 的 PyTorch 散射矩阵与自动微分思想，但删除了以下专用假设：

- 固定 774 个光谱点；
- 固定五个通道；
- 固定材料组合；
- 固定层数和 10–800 nm 厚度范围；
- 固定半无限基底。

新版支持任意波长网格、任意相干层数、多角度、多偏振、多目标、固定层与可优化层混合、每层独立厚度范围、多初值、Adam/L-BFGS、加工量化，以及平均或最差点约束。

可微后端只负责提出候选厚度。完成后，系统必须用独立 NumPy 散射矩阵重新计算；两者目标误差一致且物理审计通过，才能生成 `INDEPENDENT_VALIDATION.json` 的 `passed` 状态。

当前 Python 环境没有安装 Torch 时，启动器会依次寻找：

1. `VERITMM_PHYSICS_PYTHON` 指定的 Python；
2. `.venv-physics`；
3. 可用的外部 Python 环境。

因此材料解析和常规仿真保持轻量，只有优化任务才进入较重的物理环境。

## 论文趋势复现

公开案例采用 DOI `10.3390/s22103627` 的双 DBR 空气腔。论文给出了六对 TiO₂/SiO₂ 的实测厚度、64.4 nm TiO₂ 终止层和 600 nm 空气腔，并报告：

- 约 590–870 nm 的反射禁带；
- 639.1 nm 的窄腔模反射谷；
- 超过 145 倍的归一化场强增强。

本系统没有拿论文曲线反向拟合参数，而是直接使用论文厚度，材料色散换成本地 TiO₂/SiO₂ 数据，并把玻璃近似为 n=1.52。结果为：

- 禁带约 593.75–846.50 nm，与论文区间交并比约 0.903；
- 腔模反射谷位于 637.25 nm，与论文相差 1.85 nm；
- 最大归一化场强约 303 倍；
- 最大能量守恒误差约 `1.11e-16`。

这属于“物理趋势与关键特征复现”，不是实验曲线逐点拟合。论文的专用椭偏色散和 7 nm 粗糙层未公开完整数值，因此不应声称百分之百复现。

运行命令：

```powershell
python scripts/validate_tmm_against_pmc9147317.py
```

## 不能越界的能力

当前是平面、各向同性、一维分层模型。以下问题必须换用其他工具，不能让 TMM 假装会做：

- 光栅和周期横向结构：使用 RCWA；
- 超表面单元、复杂散射和近场三维结构：使用 FDTD/FEM；
- 各向异性张量层：需要 4×4 Berreman 等方法；
- 表面粗糙引起的漫散射：需要散射模型或实验标定；
- 热传导、对流、太阳辐照和器件温度：需要热学模型；
- 非线性、增益和时间变化材料：需要专门模型。

这种明确拒绝不是能力不足的掩饰，而是保证学术结果可信的边界。

## 关键文件

- 协议：[schemas.py](../tmm_engine/schemas.py)
- 材料注册器：[material_registry.py](../tmm_engine/material_registry.py)
- 正向工作台：[workbench.py](../tmm_engine/workbench.py)
- 可微求解器：[differentiable.py](../tmm_engine/differentiable.py)
- 厚度优化器：[optimization.py](../tmm_engine/optimization.py)
- 物理运行时发现：[physics_runtime.py](../tmm_engine/physics_runtime.py)
- 常见结构构造器：[designs.py](../tmm_engine/designs.py)
- 分析工具：[analysis.py](../tmm_engine/analysis.py)
- 标准任务入口：[run_tmm_task.py](../scripts/run_tmm_task.py)
- 材料搜索入口：[search_optical_material.py](../scripts/search_optical_material.py)
- 论文复现入口：[validate_tmm_against_pmc9147317.py](../scripts/validate_tmm_against_pmc9147317.py)

## 外部依据

- refractiveindex.info 数据库说明：https://refractiveindex.info/about
- refractiveindex.info 官方数据仓库：https://github.com/polyanskiy/refractiveindex.info-database
- 独立参考求解器 `tmm`：https://github.com/sbyrnes321/tmm
- 论文复现案例：https://doi.org/10.3390/s22103627
