# AR-OPD 交接记录

本文记录远端仓库已提交的实现，以及已经确定但尚未实现的大模型与环境主线。
代码仓库为 <https://github.com/mayiqun1213-sys/ar-opd>，分支为 `main`。
本文对应的实现基线是 `7a5e400`。

## 远端已提交实现

| Commit | 内容 |
| --- | --- |
| `6327d8b` | 初始化 Python 项目、测试配置和 Git 忽略规则。 |
| `325ef56` | 加入 step-level Student rollout、S/T/F 候选、净价值 gate、共享 actor/value toy 模型和 SMDP PPO。 |
| `406dbf4` | 加入 executed-only Corrective-SFT/Fallback-SFT、分段 replay、原子 checkpoint 和精确 resume。 |
| `25051d7` | 加入独立 Student-only rollout 上的 OPD、Teacher 全动作分布标注和 forward-KL 更新。 |
| `234e9e3` | 将环境、Teacher、候选评估器和 episode 生命周期改为结构化适配器接口。 |
| `7a5e400` | 加入可 reset/replay 的 TextWorld 边界、动态合法动作 mask、状态指纹、单 scratch 反事实回放、fake backend 和闭环测试。 |

当前 gate 没有单独 Router。每个决策点先产生 Student proposal，再比较：

- `S`：执行 Student 单步；
- `T`：执行 Teacher 单步纠正；
- `F`：执行 Teacher 短恢复。

gate 使用候选任务价值减去 Teacher 查询成本和执行成本。PPO 在 decision
boundary 上计算 duration-aware return。Teacher primitive steps 保留在轨迹中，
但不会作为 Student actor action；被 Teacher 覆盖的 Student proposal 仍保留在
PPO 决策记录中。

局部 SFT 只使用环境实际执行的 Teacher steps，未选中的 Teacher 候选和未执行的
恢复后缀不会进入 SFT。OPD 使用另一批全程只执行 Student 的新轨迹，Teacher
只做分布标注，不接管环境，标注成本与 PPO reward 分开记录。

TextWorld runtime 已实现完整 episode identity、command trace、boundary
fingerprint、terminated/truncated 分类和 reset/replay。后端调用发生异常、返回
不一致结果或 replay mismatch 后，该环境实例进入 faulted 状态，只允许关闭。

最近一次验证：

```text
PYTHONPATH=src python -W error -m unittest discover -s tests -v
Ran 111 tests ... OK

PYTHONPATH=src python -m ar_opd.fake_textworld_smoke
decisions: S, F
executed actions: 1, 2, 1
PPO update: completed
```

## 当前代码边界

当前 Student 是 `src/ar_opd/models.py` 中的小型 MLP `ActorCritic`，Teacher
是 toy/fake oracle。固定动作分类头和稳定哈希文本编码用于测试，不是大模型
训练实现。

仓库目前没有 Transformers、PEFT、Accelerate、ALFWorld 或真实
TextWorldExpress 集成；没有下载模型权重或环境数据。当前机器没有 Java，因此
`7a5e400` 只验证 dependency-free fake TextWorld backend，没有运行真实 JVM
环境。

## 大模型与环境主线

当前机器环境已经核对为：

- GPU：Tesla V100-PCIE-32GB，32,768 MiB，compute capability 7.0；
- Python：3.12.3；
- PyTorch：2.8.0+cu128；
- V100 使用 FP16，不使用 BF16。

模型组合已经确定为：

- Student：[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)；
- Teacher：[`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)。

两者使用官方 post-trained checkpoint，许可证为 Apache-2.0。项目只使用文本
输入和 non-thinking 模式。Student 的文本主干接 LoRA 和共享 scalar value
head；Teacher 冻结并只做推理。首轮运行上下文限制为 2K–4K。模型权重、缓存、
环境数据和训练 checkpoint 不进入 Git。

训练栈为 Transformers + PEFT + Accelerate，首轮不使用 Ray、vLLM 或
FlashAttention-2。Qwen3.5 接口使用 Transformers 的文本 causal-LM/hidden
state 能力；Accelerate 负责 FP16、梯度累积和 checkpoint。

正式环境主线是 ALFWorld text-only：

- 官方划分为 3,553 train、140 valid_seen、134 valid_unseen；
- episode 上限为 50 个高层文本动作；
- 每个状态提供动态 `admissible_commands`；
- 任务、观测、合法命令、实际命令和 replay identity 都按文本环境记录。

大模型策略不使用跨场景固定动作分类头。Student 对当前
`admissible_commands` 的命令文本计算条件 log-prob，并在当前候选集合中形成
策略分布；Teacher 的单步纠正和短恢复也必须落到各步的合法命令。稳定 action
identity 只用于 trajectory、fingerprint 和 replay。

TextWorldExpress fake runtime 继续承担无模型下载的回放与 S/T/F 回归测试；
ALFWorld adapter、Qwen policy/value、Qwen Teacher 和真实大模型训练入口尚未
写入仓库。

## 尚未实现的主线工作

1. 增加隔离的 LLM/ALFWorld 可选依赖并锁定实际运行版本。
2. 实现 Qwen3.5 candidate scorer、LoRA Student 和共享 value head。
3. 实现冻结的 Qwen3.5-2B Teacher 单步纠正与短恢复。
4. 实现固定 ALFWorld game 的 online/scratch reset-replay adapter。
5. 完成真实模型前向、LoRA 单步更新、同卡 Student/Teacher 驻留和峰值显存记录。
6. 打通 ALFWorld 上一次 PPO、OPD、局部 SFT 更新及 Student-only 评测。
