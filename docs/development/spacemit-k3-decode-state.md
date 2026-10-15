# K3 八线程 decode 状态算子优化（2026-10-15）

分支：`perf/k3-decode-state`。基线代码：`3da0d6cc6f5c99867cb2665fa67bafb164b2b5a4`。
模型：`Qwen3.5-0.8B-Q4_1.gguf`，SHA256 `e5926ccfef0c54aebf5d8bda01b2fb6c12ceff4f02a490bc108c5fabf60b334e`。

## 保留的改动

- **CPY**：对互不重叠的 F32 `[3, rows, 1, 1]` 源视图、16 字节行步长、连续目标，使用 RVV 三字段分段加载/存储。按整行分工，处理向量尾部，不读取每行第四字段。重叠判断包含源行间空隙；其他布局保留原路径。
- **GATED_DELTA_NET**：当 token 数不超过 4 且 head 能放入 m4 时，使用 m4 融合行更新；较宽 head 保留 m8，较长 prefill 保留原算法。目标 head=128 在 A100 上符合条件。原 m8 循环每行有三次 `vl8re32.v` 和一次 `vs8r.v`，即约 4 KiB 额外栈读写；新 m4 循环没有这些溢出读写。
- **边界修复**：CPY 和 GET_ROWS 对没有分配到元素的线程直接返回，避免短张量在 8 线程下把负数长度传给拷贝函数。新增的 CPY 单行、GET_ROWS 三列用例在基线均以 SIGSEGV 退出，候选通过。

没有改变量化、图缓存所有权、MulMat/TCM 使用的全局 `memcpy1d` 或测试误差阈值。

## 整模型实测

K3 A100 核 8–15，8 个 SPERT worker，1.8 GHz；X100 CPU 后备线程 8 个，核 0–7，2.2 GHz。两簇 governor 均为 userspace，测试未调整频率。
SpacemiT GCC 15.2.0（toolchain v1.2.4），项目 toolchain 文件使用 v1.1.2 sysroot；spine-runtime 0.6.3。Release，`GGML_SPACEMIT=ON`、`GGML_CPU_REPACK=OFF`，RVV/ZVFH/ZFH/ZICBOP/ZIHINTPAUSE/ZBA 开启。

按 baseline → candidate → candidate → baseline 顺序，每次重复 5 次；下表各包含 10 次未插桩样本，± 为样本标准差。

| 指标 | 基线 | 候选 | 变化 |
|---|---:|---:|---:|
| tg128，token/s | 31.258 ± 0.617 | 33.564 ± 0.576 | **+7.38%** |
| pp128，token/s | 212.792 ± 1.935 | 212.819 ± 2.836 | +0.01%，基本持平 |

以平均吞吐换算，decode 约从 31.99 降至 29.79 ms/token。本结果限定于本模型和 tg128 配置，不代表任意上下文长度。

```bash
export GGML_SPACEMIT_BACKEND=1 GGML_SPACEMIT_WORKERS=8
export SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15
unset SPERT_BACKEND GGML_TRACE LIGHT_PROFILE
# LD_LIBRARY_PATH 指向相应基线/候选 bin 目录
llama-bench -m Qwen3.5-0.8B-Q4_1.gguf \
  -dev SPACEMIT0 -ngl 99 -t 8 -C 0xff --cpu-strict 1 \
  -p 128 -n 128 -b 512 -ub 128 -fa on -lm none -r 5 -o json
```

## 算子收益与 GET_ROWS 实验

独立算子 perf 测试（8 worker，含算子内同步，单次测量，不能直接替代整模型结果）：

| 算子及布局 | 基线 μs/次 | 候选 μs/次 | 耗时变化 |
|---|---:|---:|---:|
| CPY，3×6144 卷积状态视图 | 82.92 | 8.75 | −89.4% |
| GDN，H=16、S=128、T=1 | 104.07 | 53.76 | −48.3% |
| CPY，连续 1 MiB 状态 | 65.87 | 66.48 | 基本持平 |
| GET_ROWS，单行 1 MiB | 66.70 | 66.69 | 基本持平 |

整模型轻量插桩另做 4 对交错测量，每次取 63 个稳态 token，各版本共 252 个 token。以下是跨 8 worker 的开始/结束包络耗时，包含算子内部同步；插桩会扰动执行，不能用其总耗时替代上面的未插桩吞吐。

| 后端算子 | 基线 ms/token | 候选 ms/token |
|---|---:|---:|
| CPY 合计，36 次/token | 3.271 | 2.093 |
| └ 卷积状态 CPY，18 次 | 1.520 | 0.312 |
| └ 连续 GDN 状态 CPY，18 次 | 1.750 | 1.781 |
| GATED_DELTA_NET，18 次 | 2.707 | 2.081 |
| SPACEMIT GET_ROWS，38 次 | 1.951 | 1.948 |

另有 CPU 输入 embedding GET_ROWS 1 次/token，插桩测得 0.053→0.207 ms，波动较大，未将其解释为后端算子变化。包含它的全部非矩阵算子合计为 10.236→8.624 ms/token。

GET_ROWS 尝试了原拷贝、libc、e8m4、e32m4、e32m8 双路展开、双路展开加预取，共 6 种实现：

- 同一 1 MiB 状态反复拷贝、命中缓存时，e8m4 的中位耗时由原实现的 27.38 降至 18.32 μs。
- 轮转 18 份状态（源+目标共 36 MiB）时，原实现 68.53 μs、e8m4 69.50 μs、e32m8 67.24 μs、预取版 67.14 μs，读写总带宽约 30–31 GB/s。
- 将 e8m4 接入 GET_ROWS 和连续 CPY 后，整模型两组 tg128 均值为 33.668、33.592 token/s，与保留版本的测量波动重合。因此撤回这个替代实现，只保留边界修复。

GET_ROWS 的主要剩余成本仍是状态数据搬运；这轮局部复制循环调整没有建立可重复的整模型收益。减少状态副本需要另外验证图与序列缓存的别名/生命周期语义，本分支未尝试。

## 正确性与限制

- 原有相关算子集：基线 111/124 通过，候选加上新增用例后 135/148 通过；失败集合都是同样的 13 个原有 CPY 转置/置换相关用例，无新增失败。不能宣称完整测试集全绿。
- 共新增 25 个用例，其中 24 个执行并通过；另 1 个 I32 GET_ROWS 在 SPACEMIT0 上不受支持而跳过。覆盖目标状态布局、短线程分区、尾部、平面回退、自拷贝、GET_ROWS 视图/索引/批次、GDN 128/256 维、KDA、多序列与快照。GDN 共 40/40 通过 CPU 参考比较。
- 直接 GDN 连续状态更新对比：128/256 维、标量门控/KDA 共 4 种情况，每种连续更新 64 步；每步 attention 输出和每 8 步保存的完整状态均与基线逐位一致（差异元素数 0）。输入哈希一致，所有输出有限。该验证隔离了整模型其他算子的运行不确定性，但不代表所有输入均已穷举。
- 整模型 teacher forcing 使用英文说明、算术、中文说明三个 prompt，各固定 64 步，每 8 步保存完整 logits。候选相对基线 top-1 为 188/192，logits RMS 差 0.17891；**同一个基线重跑也只有 184/192 一致，RMS 差 0.15964**，并在同一个初始 prefill 输出位置出现差异。因此整模型存在基线运行不确定性，不能宣称逐 token / logits 一致，也不能用仅一次基线重跑建立可靠误差上界。
- 源码及边界条件已完成独立审查；没有发现本次新增缺陷。重叠视图保留原实现行为，不为任意重叠 CPY 新增 memmove 语义。

## 复现产物

工作区根目录下 `artifacts/k3-opt/` 保留了构建、原始结果和实验工具：

- `results/bench-summary.json` 以及原始 `bench-*.json`：整模型采样。
- `results/tests-original.log`、`tests-candidate.log`、`tests-comparison.json`：正确性与原有失败集合。
- `profile/{baseline,candidate}/`：算子统计、原始二进制轨迹、各次测量耗时。
- `copy-probe/`、`logits-check/`、`gdn-check/`：独立实验源码、构建命令、文件格式和比较工具。
- `results/loader-*.txt`、`hashes-*.txt`：运行库解析与哈希；`asm/`：汇编证据。
- `run-ab.sh`、`run-copyvariant.sh`、`run-profile.sh`：真机测量脚本。

最终保留的 `libggml-spacemit.so.0.26.0` SHA256：`f4cb7ab8b52c8497c9906bd7db53daed7961e84dbca884bfbdef676db67c2967`。撤回拷贝实验后重新构建，二进制与已测候选一致。
