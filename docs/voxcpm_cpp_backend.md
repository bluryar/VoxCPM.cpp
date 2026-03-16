# VoxCPM.cpp Vulkan 首轮验证说明

## 目标

这一轮只做两件事：

1. 给 VoxCPM.cpp 接上 Vulkan 单后端初始化路径
2. 保持 ARM64 CPU `GGML_NATIVE=ON` 的本地优化能力

这一轮不做 `ggml_backend_sched` 多后端调度重构。当前项目的单后端执行模型
`ggml_gallocr + ggml_backend_graph_compute` 已经足够验证 "Vulkan 能不能跑起来"。

## 当前项目状态

截至 2026-03-15，项目现状如下：

- `VoxCPM.cpp/CMakeLists.txt` 之前没有把 Vulkan 选项透传给内置 ggml
- `VoxCPM.cpp/src/backend.cpp` 之前只会初始化 CPU backend
- `VoxCPM.cpp/examples/voxcpm_tts.cpp` 之前把 backend 硬编码成 `BackendType::CPU`
- `VoxCPM.cpp/third_party/ggml` 已经包含 Vulkan/CUDA/HIP/Metal/SYCL 后端源码

本次代码改动后：

- 新增 `VOXCPM_VULKAN` 和 `VOXCPM_NATIVE` CMake 选项
- `VoxCPMBackend` 现在支持 `CPU` / `Vulkan` / `Auto`
- `voxcpm_tts` 新增 `--backend cpu|vulkan|auto`

## 首轮范围

### 已实现

- CMake 可透传 `VOXCPM_VULKAN` 到 ggml 的 `GGML_VULKAN`
- CMake 可透传 `VOXCPM_NATIVE` 到 ggml 的 `GGML_NATIVE`
- `VoxCPMBackend` 会通过 `ggml_backend_dev_count()` 枚举设备
- `--backend vulkan` 会只接受 Vulkan registry 下的 GPU 设备
- `--backend auto` 会优先尝试 Vulkan，失败时回退到 CPU
- 启动日志会打印最终实际使用的 backend 和设备描述

### 本轮刻意不做

- 不做 `ggml_backend_sched_t`
- 不做 CUDA/HIP/SYCL/Metal 接线
- 不强绑 `GGML_CPU_ARM_ARCH`
- 不承诺 Vulkan 一定比 CPU 更快

## 系统前置条件

在这台 Orange Pi 6 Plus 上，代码是否能真正跑 Vulkan，不只取决于项目代码，还取决于系统 Vulkan
环境。

Vulkan 构建和运行前，必须同时满足：

1. `vulkaninfo` 可以成功创建设备实例并列出 GPU
2. `glslc` 在 `PATH` 中可执行

当前这台机器实际检查结果是：

- 系统中有 `libvulkan`
- `vulkaninfo` 目前失败，报错包含：
  - `Found no drivers`
  - `vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER`
- `glslc` 当前不在 `PATH`

所以当前状态下：

- 可以把 VoxCPM 的 Vulkan 接口接好
- 但还不能直接证明 Vulkan 推理已经可用

要真正进入 Vulkan 构建验证，必须先修好系统环境。

## 为什么不需要手动额外链接 ggml-vulkan

内置 ggml 子树的 `ggml_add_backend_library(...)` 会把启用的 backend 库挂到 `ggml`
目标上。

因此在 VoxCPM 项目层：

- 需要做的是在 `add_subdirectory(third_party/ggml ...)` 之前设置 `GGML_VULKAN`
- 不需要再手动 `target_link_libraries(voxcpm PUBLIC ggml-vulkan)`

## 构建方式

### CPU 构建

```bash
cd /home/orangepi/Codes/ggbond/VoxCPM.cpp
cmake -B build -DVOXCPM_NATIVE=ON
cmake --build build
```

### Vulkan 构建

使用单独目录，避免覆盖当前可工作的 CPU `build/`：

```bash
cd /home/orangepi/Codes/ggbond/VoxCPM.cpp
cmake -B build-vulkan -DVOXCPM_VULKAN=ON -DVOXCPM_NATIVE=ON
cmake --build build-vulkan
```

如果这一步失败，优先检查的不是 VoxCPM 代码，而是：

- `vulkaninfo`
- `glslc --version`
- 系统 Vulkan driver / ICD 是否正确安装

## 运行方式

### CPU

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build/examples/voxcpm_tts \
    --backend cpu \
    --text "测试一下，这是一个流式音频" \
    --prompt-audio /home/orangepi/Codes/ggbond/examples/dabin.wav \
    --prompt-text "可哪怕位于堂堂超一品官职,在十 二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应" \
    --output /home/orangepi/Codes/ggbond/voxcpm_stream_single_final.wav \
    --model-path /tmp/voxcpm1.5-iq3xxs.gguf \
    --threads 8 \
    --inference-timesteps 10 \
    --cfg-value 2.0
```

### Vulkan

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build-vulkan/examples/voxcpm_tts \
    --backend vulkan \
    --text "测试一下，这是一个流式音频" \
    --prompt-audio /home/orangepi/Codes/ggbond/examples/dabin.wav \
    --prompt-text "可哪怕位于堂堂超一品官职,在十 二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应" \
    --output /home/orangepi/Codes/ggbond/voxcpm_stream_single_final.wav \
    --model-path /tmp/voxcpm1.5-iq3xxs.gguf \
    --threads 8 \
    --inference-timesteps 10 \
    --cfg-value 2.0
```

### Auto

```bash
/home/orangepi/Codes/ggbond/VoxCPM.cpp/build-vulkan/examples/voxcpm_tts \
    --backend auto \
    --text "测试一下，这是一个流式音频" \
    --prompt-audio /home/orangepi/Codes/ggbond/examples/dabin.wav \
    --prompt-text "可哪怕位于堂堂超一品官职,在十 二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应" \
    --output /home/orangepi/Codes/ggbond/voxcpm_stream_single_final.wav \
    --model-path /tmp/voxcpm1.5-iq3xxs.gguf \
    --threads 8 \
    --inference-timesteps 10 \
    --cfg-value 2.0
```

## 运行语义

### `--backend cpu`

- 永远使用 CPU
- 保持和当前已有命令兼容

### `--backend vulkan`

- 只尝试 Vulkan backend
- 找不到 Vulkan 设备或 Vulkan 初始化失败时直接报错
- 不允许静默回退到 CPU

### `--backend auto`

- 优先尝试 Vulkan
- Vulkan 不可用时回退到 CPU
- 启动日志会打印最终实际 backend

### `GGML_DISABLE_VULKAN=1`

ggml 自带 Vulkan runtime disable 开关：

```bash
GGML_DISABLE_VULKAN=1 /home/orangepi/Codes/ggbond/VoxCPM.cpp/build-vulkan/examples/voxcpm_tts --backend auto ...
```

这个场景下应该稳定回退到 CPU。

## ARM64 优化策略

这一轮默认只启用：

```bash
-DVOXCPM_NATIVE=ON
```

原因：

- 你的 CPU 已经具备 `dotprod` / `i8mm` / `sve` / `bf16` 等特性
- ggml 会在 native 构建下尽量针对当前机器生成更合适的代码
- 先不要在首轮把问题复杂化到手动指定 `GGML_CPU_ARM_ARCH`

等 Vulkan 基线跑通后，再单独评估是否继续尝试：

```bash
-DGGML_CPU_ARM_ARCH=...
```

## 验证清单

### 系统前置验证

```bash
vulkaninfo
glslc --version
```

两个命令都必须成功。

### 配置验证

```bash
cmake -B build-vulkan -DVOXCPM_VULKAN=ON -DVOXCPM_NATIVE=ON
```

### 应用验证

1. `--backend cpu` 能正常运行
2. `--backend vulkan` 能打印 Vulkan 设备名并跑通
3. `GGML_DISABLE_VULKAN=1 --backend auto` 能回退到 CPU
4. `--backend vulkan` 在 Vulkan 不可用时快速失败

### 性能记录

使用同一模型、同一输入，记录：

- CPU 总耗时
- Vulkan 总耗时

如果 Vulkan 慢于 CPU，也按事实记录。

## 监控 Vulkan / ARM GPU 占用（类似 htop）

在 Linux 上查看 Vulkan 使用的 ARM GPU（如 Mali-G720）占用率，可用下面几种方式。

### 1. MangoHud 叠加层（推荐，适合 Vulkan 进程）

对 Vulkan 应用直接加一层实时叠加，显示 GPU/CPU 占用、帧率等：

```bash
# 安装（示例：Debian/Ubuntu）
sudo apt install mangohud

# 运行 voxcpm_tts 时启用叠加
MANGOHUD=1 /path/to/voxcpm_tts --backend vulkan --model-path ... --text "测试" --output out.wav ...
```

若只关心 GPU，可缩小显示项：

```bash
MANGOHUD_CONFIG="cpu_stats=0,fps=0" MANGOHUD=1 /path/to/voxcpm_tts --backend vulkan ...
```

注意：MangoHud 的 GPU 占用来自 Vulkan 层，在部分 ARM/Mali 驱动上可能不显示或不准，需本机实测。

### 2. 系统 devfreq/sysfs（通用 ARM GPU）

很多 ARM SoC 的 GPU 通过 devfreq 暴露负载，可在**另一个终端**用下面命令看实时占用（数值多为 0–100 或 0–1024 等，视内核而定）：

```bash
# 查看是否有 GPU 相关 devfreq 设备
ls /sys/class/devfreq/

# 若有类似 ff9a0000.gpu 或 fdab0000.gpu 的目录，可轮询 load 或 utilization
watch -n 0.5 'cat /sys/class/devfreq/*/load 2>/dev/null; cat /sys/class/devfreq/*/utilization 2>/dev/null'
```

部分板子路径可能是 `/sys/devices/platform/.../utilization` 或 `.../cur_freq`，可用 `find /sys -name utilization -o -name load 2>/dev/null` 搜一下。

### 3. nvtop（若支持你的 GPU）

nvtop 是类 htop 的 GPU 监控工具，支持 AMD/Intel/NVIDIA/Qualcomm 等，**目前对 ARM Mali 支持有限**，可先试装看是否有你的设备：

```bash
sudo apt install nvtop
nvtop
```

若 nvtop 里看不到 Mali 或 Vulkan 设备，则用上面 1 或 2。

### 4. 进程级 GPU 占用（Panfrost 等）

若使用 Panfrost 等内核驱动，有时可通过进程 fdinfo 看到 GPU 使用（需内核开启相应统计）：

```bash
# 找到 voxcpm_tts 的 PID 后
cat /proc/<PID>/fdinfo/* 2>/dev/null | grep -i drm
```

总结：**优先用 MangoHud 看 Vulkan 进程的叠加数据；同时用 devfreq/sysfs 看整机 GPU 占用（类似 htop 看整机 CPU）。**

## 后续第二阶段

如果第一阶段确认 Vulkan 可用，再进入下一阶段：

- `ggml_backend_sched_t`
- CPU/GPU 混合调度
- 其他后端接线：CUDA / HIP / SYCL / Metal

但这些都不应该阻塞这一轮的 Vulkan 首轮验证。
