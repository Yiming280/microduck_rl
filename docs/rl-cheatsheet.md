# Microduck RL 速查表 (Cheatsheet)

> 双足机器人 Microduck(800g / 25cm / 14 × XL330 舵机)的强化学习训练与推理速查。
> 技术栈:**mjlab**(MuJoCo Warp 物理引擎)+ **rsl_rl**(PPO)。50 Hz 控制,训练产物导出 ONNX 部署到真机。
> 更多细节见根目录 [AGENTS.md](../AGENTS.md)。

---

## 1. RL 原理速记

### 1.1 训练大循环(每 iteration)

```
循环 (max_iterations 次):
  1. rollout:   策略在 num_envs 个并行环境里各走 num_steps_per_env=24 步,
                收集 (obs, action, reward, next_obs, done) 存入 buffer
  2. compute_returns: 用 value function 算 advantage 和 return(GAE,见 1.4)
  3. update:    把 buffer 切成 num_mini_batches 个小批,
                每小批过 num_learning_epochs 遍,梯度下降最小化 PPO loss(见 1.5)
```

每 iteration 总样本数 = `num_envs × 24`。默认 4096 envs ≈ 9.8 万条。

### 1.2 两个网络:actor 与 critic(value function)

| | actor(策略) | critic(价值函数 V) |
|---|---|---|
| 输入 | 61D 观测 | 61D 观测 |
| 输出 | 高斯分布(14 个舵机动作的均值+方差),动作采样 | **1 个标量 V(s)** |
| 结构 | MLP `hidden_dims=(512,256,128)` | 同结构,最后一层输出 1 维 |
| 作用 | 决定"怎么动" | 估计"长期能拿多少分" |

**价值函数** `V(s)` = 从状态 s 出发、按当前策略继续走,未来所有奖励的(折扣)累加期望。

### 1.3 reward 怎么定义和计算

- **定义**:每个 reward term 是 [mdp.py](../src/mjlab_microduck/tasks/mdp.py) 里的一个函数,输入环境、返回 `(num_envs,)` 张量。
- **装配**:在 env cfg 里挂权重,如 `cfg.rewards["upright"].weight = 2.0`、`cfg.rewards["foot_slip"].weight = -0.1`。
- **计算**:`总奖励 = Σᵢ termᵢ(env) × weightᵢ × dt`(dt=0.02s @50Hz)。

> **符号约定(踩坑重灾区)**:mjlab 自带代价函数返回 ≥0 → 配**负**权重;本项目 `*_penalty`/`*_l1` 返回 ≤0 → 配**正**权重。配反 = 负负得正 = 把违规当奖励,策略会钻空子。

### 1.4 advantage 与 return(GAE)

```python
delta_t  = r_t + γ·(1-done)·V(s_{t+1}) − V(s_t)   # TD error
A_t      = delta_t + γ·λ·(1-done)·A_{t+1}         # GAE 递推
return_t = A_t + V(s_t)                             # critic 的训练目标
# advantage 最后标准化到均值 0、方差 1
```

参数:`γ=0.99`(折扣)、`λ=0.95`(GAE)。

### 1.5 PPO loss(三项相加)

```python
loss = surrogate_loss + value_loss_coef · value_loss − entropy_coef · entropy

# ① 策略(actor)surrogate loss —— PPO 的 clip 下界
ratio = exp(logπ_new − logπ_old)
surrogate_loss = max(−A·ratio, −A·clamp(ratio, 1−ε, 1+ε)).mean()   # ε=0.2

# ② 价值(critic)loss —— MSE
value_loss = (V(s) − return)²

# ③ entropy 正则(负号 = 奖励探索)
− entropy_coef · entropy.mean()
```

关键超参:`clip_param=0.2`、`value_loss_coef=1.0`、`entropy_coef=0.01`、`desired_kl=0.01`(自适应学习率)。

### 1.6 多任务怎么兼容:注册表 (registry) 模式

- 每个任务在 [tasks/__init__.py](../src/mjlab_microduck/tasks/__init__.py) 里调一次
  `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`。
- 训练脚本 `train <TASK_ID>` 只按 ID 查表拿到三个 cfg,循环代码对任何任务都一样。
- `uv run list-envs` 列出所有 task_id。

---

## 2. 训练命令

```bash
uv run list-envs                                      # 列出所有任务 ID
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — 长训练前必跑
uv run train <TASK_ID> --env.scene.num-envs 4096                          # 正式训练
uv run train <TASK_ID> --hf-jobs                                         # 提交到 Hugging Face Jobs
```

### 2.1 8GB 显存不 OOM(关键配置)

默认 4096 envs 在 8GB 上基本必 OOM,`num_envs` 是最大头:

```bash
# 推荐:1024 envs + 8 minibatches(已验证 RTX 4060 Laptop 8GB 跑完 4000 iters)
uv run train <TASK_ID> \
  --env.scene.num-envs 1024 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.max_iterations 4000
```

- **降 `num_envs`**(4096→1024/2048):物理状态显存线性缩放,最大头。
- **升 `num_mini_batches`**(4→8/16):降 update 阶段峰值显存。
- 还不够 → 改 cfg 里 actor/critic 的 `hidden_dims=(512,256,128)` → `(256,128,64)`。
- 别动 `num_steps_per_env=24`(和课程表时序强绑定)。
- 降 envs 后吞吐变小,想凑同样本量要等比放大 `max_iterations`。

监控 GPU usage
```
watch -n 1 nvidia-smi
```

### 2.2 续训 / 读 run

```bash
# 续训(本地 checkpoint)
uv run train <TASK_ID> --agent.load-checkpoint model_4000.pt --agent.resume True

# 或显式用 --agent.load-run 锁定某次训练，如:
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.max_iterations 2000 \
  --agent.load-run 2026-09-28_14-06-46_velocity \
  --agent.load-checkpoint model_3999.pt \
  --agent.resume True
# 想确认有哪些 run 目录,先 ls logs/rsl_rl/velocity/ 看一眼。

#更稳的做法:用 --wandb-run-path 续训(它唯一标识 entity/project/run_id,无歧义)，如：
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.algorithm.num-mini-batches 8 \
  --agent.max_iterations 2000 \
  --wandb-run-path weiyiming280-technical-university-of-munich/mjlab_microduck/ktonsxzo \
  --wandb-checkpoint-name model_3999.pt \
  --agent.resume True

# 读 wandb:每项 Episode_Reward/<penalty> 必须 ≤ 0;主任务 term 要真在涨
# wandb project = mjlab_microduck;本地日志 logs/<experiment_name>/
```

---

## 3. 测试 / 推理命令

### 3.1 同步 offline 日志(在没网环境训练过才需要)

```bash
uv run wandb sync wandb/offline-run-<时间戳>-<runid>
uv run wandb login        # 首次需要,粘贴 API key
```

`--wandb-run-path` 格式:`<entity>/<project>/<run_id>`(如 `weiyiming280-technical-university-of-munich/mjlab_microduck/ktonsxzo`)。

### 3.2 可视化测试(在 sim viewer 里看策略)

```bash
uv run play <TASK_ID> --wandb-run-path <entity>/<project>/<run_id>
# 指定某个 checkpoint:
uv run play <TASK_ID> --wandb-run-path <...> --wandb-checkpoint-name model_4000.pt
```

> 注意:play 在训练环境里跑,会自动套 obs normalizer,看不出"normalizer 没烤进 ONNX"的 bug,只适合看效果。

### 3.3 导出 ONNX(部署格式,**必须走这条路**)

```bash
uv run scripts/export.py <TASK_ID> --wandb-run-path <...> --checkpoint 4000 --onnx-file output.onnx
```

- `--checkpoint 4000` 选第 4000 次迭代模型(不写取最新)。
- normalizer 在这一步烤进 ONNX。**永远不要手搓 checkpoint 转 ONNX。**
- ⚠️ wandb 上训练时自动存的那个 `*_velocity.onnx` 是开局 model_0 的初始导出,别当最终策略。

### 3.4 CPU 部署预演(最接近真机)

```bash
uv run scripts/infer_policy.py --walking output.onnx --new-cmd-obs
```

- **必须加 `--new-cmd-obs`**:默认是旧版 51D obs(3D twist),训练用的是 61D(13D 指令块)。不加会报
  `Got invalid dimensions for input: obs ... Got: 51 Expected: 61`。
- 其他任务用对应 flag:`-s/--standing`、`--sitstand`、`--kick-left/right`、`--roulade`(后三者也要 `--new-cmd-obs`)。
- 默认 BAM M6 执行器(与训练一致);`--no-bam` 退化为 XML PD。
- 命令槽位:全零 twist = "站住",别误判成"策略无视按键"。

### 3.5 发布给真机(上机器人时才需要)

```bash
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint 4000 \
  --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
```

### 3.6 跑测试

```bash
uv run --with pytest pytest tests/      # CPU,验证配置/奖励函数回归
```

---

## 4. 必背的坑 (Invariants)

- **obs 布局 61D 且全家共享**:48 基础本体感 + 13D 指令块 `[twist(3), head_pose(4), body_pose(6)]`。不用的指令槽**零填充,不删除**。
- **执行器是 BAM**(电压控制):standalone cfg 要注册 `expand_bam_friction_fields`;关节摩擦 DR 要调 `friction_scale` 而非 `dof_frictionloss`(BAM 下被清零)。
- **obs normalization 开启** → normalizer 必须烤进 ONNX,只用 `scripts/export.py`。
- **策略无滤波**:别乱加 EMA 低通,除非有配套 runtime flag + transfer 测试。
- **DR 不能跨 reset 累积**:mjlab 1.3.0 的 `dr.*` add/scale 天然不累积;自定义 DR 要 restore-then-apply。
- **reward 符号**:每个 `Episode_Reward/<penalty>` 在 wandb 里必须 ≤ 0。
- **被动关节命名 `passive_*`**:选择器用 `^(?!passive_).*`;新增 `passive_` 正则要精确(如 `^passive_.*wheel`),别误伤 backlash 关节。
- **`-Backlash-` 变体必须 mirror 基任务的机器人模型**,保证 A/B 对比不混杂。

---

## 5. 一个完整数据流

```
task_id ─查表─▶ env_cfg(机器人+观测+奖励+事件) + rl_cfg(actor/critic/PPO 超参)
   │
   ├─ env.step(): 物理仿真一步 → 61D 观测 + 标量 reward(= Σ weightᵢ·termᵢ·dt)
   │
   └─ 24 步后:
        critic 算 V(s) → GAE 算 advantage/return
        actor 损失: clip 后的 surrogate loss
        critic 损失: (V−return)²
        总 loss = surrogate + value − entropy·熵
        反向传播更新两网络 → 下一 iteration
```

**训练完成后的流程**:`play` 看效果 → `export.py` 出 ONNX → `infer_policy.py --new-cmd-obs` 真机排练 → `publish` 上真机。
