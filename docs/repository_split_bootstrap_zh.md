# VoxCPM.cpp 独立仓库初始化与 ggml subtree 导入记录

## 文档目的

本文档记录 `VoxCPM.cpp` 从 `ggbond` 仓库子目录转换为独立 Git 仓库的实际执行过程，用于后续追溯：

- 当时的背景和目标
- 与原始计划的差异
- 实际执行的命令和结果
- 当前仓库状态
- 后续仍需完成的开源整理项

## 背景

执行时的目录关系如下：

- `ggbond` 是一个已有 Git 仓库
- `VoxCPM.cpp` 是 `ggbond` 下的一个子目录
- `VoxCPM.cpp` 当时自身没有独立 `.git/`
- `VoxCPM.cpp/third_party/ggml` 是一个单独 clone 的 `ggml` 仓库
- `VoxCPM.cpp/third_party/json`
- `VoxCPM.cpp/third_party/llama.cpp`
- `VoxCPM.cpp/third_party/whisper.cpp`
- `VoxCPM.cpp/third_party/SenseVoice.cpp`

目标是把 `VoxCPM.cpp` 转为独立 Git 仓库，并将 `third_party/ggml` 纳入可维护的 subtree 形态，便于后续同步上游并保留本地补丁。

## 参考计划与校验结论

参考文档：

- `/home/orangepi/.claude/plans/polished-jumping-firefly.md`

该计划的大方向是正确的，但在实际执行前发现有几处需要修正：

### 1. `VoxCPM.cpp` 并非完全脱离 Git

原计划将 `VoxCPM.cpp` 描述为“当前不是 Git 仓库”，这在局部上是正确的，但更准确的描述是：

- `VoxCPM.cpp` 不是独立 Git 仓库
- 但它已经位于父仓库 `ggbond` 的工作树中

这个差异会影响后续判断，例如哪些文件已经被父仓库跟踪、哪些操作会与父仓库产生交叉影响。

### 2. `json` 的 FetchContent 回退其实已存在

执行前检查 [CMakeLists.txt](/home/orangepi/Codes/ggbond/VoxCPM.cpp/CMakeLists.txt) 发现：

- 当前工程优先使用 `third_party/json`
- 若该目录不存在，则已经自动回退到 `FetchContent`

因此原计划中“必须修改 `CMakeLists.txt` 才能支持 FetchContent”并不准确。后续如果决定移除本地 `json` clone，只需要保持现有逻辑即可。

### 3. 现有 `ggml` clone 不能直接用于 subtree add

执行前检查发现：

- `third_party/ggml` 是一个浅克隆仓库
- 本地有未提交修改：`src/ggml-vulkan/ggml-vulkan.cpp`
- 直接执行 `git subtree add --prefix=third_party/ggml ... master --squash` 会失败

失败原因不是 `subtree` 思路本身有问题，而是这个本地 clone 的历史不完整，导致 `git subtree add` 在抓取分支或提交时被浅历史限制。

## 执行前检查结果

### 工程状态

- `VoxCPM.cpp` 当时没有独立 `.git/`
- 但 `git rev-parse --is-inside-work-tree` 返回 `true`，说明它位于父仓库 `ggbond` 的工作树内

### `ggml` 状态

- 本地分支：`master`
- 本地基线提交：`4773cde162a55f0d10a6a6d7c2ea4378e30e0b01`
- 远程：`https://github.com/ggerganov/ggml.git`
- 未提交修改：
  - `src/ggml-vulkan/ggml-vulkan.cpp`

### 目录大小

执行前大致大小如下：

- `third_party/ggml`: 40M
- `third_party/json`: 257M
- `third_party/llama.cpp`: 441M
- `third_party/whisper.cpp`: 70M
- `third_party/SenseVoice.cpp`: 34M

## 实际执行过程

### 步骤 1：备份 `ggml` 本地修改与来源信息

先将 `ggml` 的本地补丁和来源信息备份到临时目录：

- `/tmp/voxcpm-bootstrap/ggml-vulkan-compat.patch`
- `/tmp/voxcpm-bootstrap/ggml-base-commit.txt`
- `/tmp/voxcpm-bootstrap/ggml-origin-url.txt`

目的：

- 避免在仓库拆分过程中丢失本地 `ggml` 修改
- 记录当时的上游来源和基线提交，便于日后回溯

### 步骤 2：更新仓库元数据文件

实际修改了以下文件：

- [.gitignore](/home/orangepi/Codes/ggbond/VoxCPM.cpp/.gitignore)
- [README.md](/home/orangepi/Codes/ggbond/VoxCPM.cpp/README.md)

其中 `.gitignore` 的策略是：

- 忽略构建产物：`build/`、`build-vulkan/`、`Testing/`
- 忽略本地环境目录：`.vscode/`、`.idea/`、`.claude/`
- 忽略模型目录：`models/`
- 忽略测试痕迹数据：`tests/fixtures/trace/`
- 保留 `third_party/ggml` 作为仓库内容
- 忽略其余参考仓库：
  - `third_party/json/`
  - `third_party/llama.cpp/`
  - `third_party/whisper.cpp/`
  - `third_party/SenseVoice.cpp/`

这里的决策是“先不删除本地参考目录，只让新仓库忽略它们”，这样不会破坏现有开发环境，也不会在初始化仓库时把这些大型参考仓库一并纳入版本历史。

### 步骤 3：初始化独立 Git 仓库

在 `VoxCPM.cpp` 目录下执行：

```bash
git init -b main
```

然后先做基础提交，但显式排除 `third_party/ggml`：

```bash
git add . ':(exclude)third_party/ggml'
git commit -m "chore: bootstrap standalone VoxCPM.cpp repository"
```

这样做的原因是：

- 如果直接把现有 `third_party/ggml` 作为嵌套仓库纳入，会留下错误的 gitlink / submodule 式结构
- 先建立一个不含 `ggml` 的基础仓库，再用 `subtree` 导入，历史结构更干净

### 步骤 4：清理首个提交中的本地生成文件

初始化后发现 `scripts/__pycache__/convert_voxcpm_to_gguf.cpython-311.pyc` 被纳入了首个提交。

因此补做了一次整理提交：

```bash
git rm --cached scripts/__pycache__/convert_voxcpm_to_gguf.cpython-311.pyc
git add .gitignore
git commit -m "chore: ignore local generated artifacts"
```

### 步骤 5：将现有 `ggml` clone 临时移出工作树

为了避免把它当作嵌套仓库错误纳入，先执行了物理挪移：

```bash
mv third_party/ggml /tmp/voxcpm-bootstrap/ggml-repo
```

### 步骤 6：处理浅克隆问题

原本尝试直接以该本地 clone 执行 subtree 导入，但失败了。

根因：

- `/tmp/voxcpm-bootstrap/ggml-repo` 是浅克隆
- `git subtree add` 依赖可用的提交历史

因此采用折中方案：

1. 将当前 `ggml` 工作树完整复制到新的临时目录
2. 移除其中原有 `.git/`
3. 用当前工作树快照重新初始化一个本地 Git 仓库
4. 从这个“正规化快照仓库”导入 subtree

这样做的效果是：

- 保留了当前 `ggml` 目录内容
- 保留了当时的本地 Vulkan 修复
- 避免被浅克隆历史阻塞

代价是：

- 这次 subtree 导入的内部 squash 来源是“本地快照仓库”
- 不是直接从 GitHub 上游完整历史抓取而来

但这不会妨碍后续继续整理上游关系，因为：

- 原始上游 URL 已记录
- 原始基线提交已记录
- 当前目录内容已成功纳入主仓库历史

### 步骤 7：导入 `ggml` subtree

最终成功执行的方式是：

```bash
git subtree add --prefix=third_party/ggml /tmp/voxcpm-bootstrap/ggml-subtree-src main --squash
```

执行完成后：

- `third_party/ggml` 已经成为当前仓库的一部分
- 其目录下不再保留独立 `.git/`

### 步骤 8：确认 Vulkan 本地修复已包含

原计划中还打算在 subtree 导入后再额外应用一次补丁。

实际比对后发现：

- 导入的 `ggml` 快照本身已经包含 `src/ggml-vulkan/ggml-vulkan.cpp` 的本地修改
- 因此没有必要再创建单独的“reapply patch”提交

## 当前仓库状态

执行完成后的提交历史：

```text
d871c1d (HEAD -> main) Merge commit '286b032928468b3272688c6a1b819532c7be14ea' as 'third_party/ggml'
286b032 Squashed 'third_party/ggml/' content from commit 9df8615
4679c4c chore: ignore local generated artifacts
e130a7b chore: bootstrap standalone VoxCPM.cpp repository
```

执行完成后的状态：

- 当前分支：`main`
- `git status --short --branch` 显示工作树干净
- `third_party/ggml` 下已无嵌套 `.git/`

## 当前结果的含义

目前 `VoxCPM.cpp` 已经完成“独立仓库化”的核心部分：

- 有了自己的 `.git/`
- 有了自己的提交历史
- `ggml` 已作为仓库内容纳入，而不是保留成外部嵌套仓库
- 其余大型参考仓库不会进入新仓库版本历史

这意味着后续可以独立进行：

- 新建远程仓库并推送
- 补 `LICENSE`
- 补更完整的 `README`
- 做 CI
- 决定是否清理本地参考目录

## 本次没有执行的事项

以下事项本次没有执行，仍属于后续开源整理工作：

- 没有新增 `LICENSE`
- 没有删除本地参考目录
- 没有把 `third_party/json` 从磁盘移除
- 没有运行完整构建和测试验证
- 没有为新仓库设置远程地址
- 没有建立 GitHub Actions CI

## 后续建议

建议按下面顺序继续整理：

1. 补充许可证文件，并确认是否继续使用 GPL v3
2. 完善 [README.md](/home/orangepi/Codes/ggbond/VoxCPM.cpp/README.md)
3. 运行一次完整构建和测试，确认拆分后无路径问题
4. 决定是否删除本地参考目录：
   - `third_party/json`
   - `third_party/llama.cpp`
   - `third_party/whisper.cpp`
   - `third_party/SenseVoice.cpp`
5. 若要继续长期维护 `ggml`，补一份本仓库内的 patch registry 文档
6. 创建远程仓库并推送

## 相关文件

- [README.md](/home/orangepi/Codes/ggbond/VoxCPM.cpp/README.md)
- [.gitignore](/home/orangepi/Codes/ggbond/VoxCPM.cpp/.gitignore)
- [CMakeLists.txt](/home/orangepi/Codes/ggbond/VoxCPM.cpp/CMakeLists.txt)
- [docs/ggml_subtree_maintenance_strategy.md](/home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/ggml_subtree_maintenance_strategy.md)
- [docs/audio_vae_ggml_operator_investigation_zh.md](/home/orangepi/Codes/ggbond/VoxCPM.cpp/docs/audio_vae_ggml_operator_investigation_zh.md)

