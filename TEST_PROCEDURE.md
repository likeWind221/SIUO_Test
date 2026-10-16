# SIUO 测试流程

以下命令均在 `siuo_test_package/` 目录执行。本交付包已包含合并后的 GRPO step 529 模型，无需再合并训练检查点。默认模型名为 `qwen25vl7b-grpo-step529`，测试集有 167 道题，每题生成 3 个回答。

## 1. 创建环境

**做什么：**创建独立的 Python 环境，避免与训练环境中的其他依赖冲突。

```bash
conda create -n siuo-test python=3.12 -y \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate siuo-test

```

预期 Python 版本为 3.12。环境包及参考版本见 `environment.txt`。

## 2. 安装依赖

**做什么：**`requirements.txt` 已包含 PyTorch、视觉依赖、模型推理和 API 判分所需的包，并已配置清华镜像。在 Linux x86_64 上，锁定的 PyTorch 2.8.0 会安装 CUDA 12.8 依赖。若目标机器驱动不支持该版本，需要调整 PyTorch 版本。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

预期 `pip check` 输出 `No broken requirements found.`。

## 3. 检查 GPU、模型和数据

**做什么：**确认模型权重和 167 道题的图片均已放到位，并查看当前节点的 CUDA 状态。

```bash
python - <<'PY'
import json
from pathlib import Path

import torch

root = Path.cwd()
items = json.loads((root / "data/siuo_gen.json").read_text(encoding="utf-8"))
weights = list((root / "model/huggingface_step_529").glob("model-*.safetensors"))
missing = [item["image"] for item in items if not (root / "data/images" / item["image"]).is_file()]
print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("questions:", len(items), "weight shards:", len(weights), "missing images:", len(missing))
assert len(items) == 167 and len(weights) == 4 and not missing
PY
```

预期输出：167 道题、4 个权重分片、0 张缺失图片。登录节点显示 `CUDA available: False` 是正常的；GPU 作业会在计算节点再次检查 CUDA，届时应为 `True`。

登录节点通常没有 GPU。在登录节点运行以下提交脚本，脚本会向集群调度器申请 GPU：

```bash
bash scripts/submit_siuo_generate.sh
```

该作业会在 `gpu_6000ada` 队列申请 1 张 GPU，在计算节点激活 `siuo-test` 环境并完成第 5 步的全量推理，不调用判分 API。日志位于 `outputs/logs/siuo-gen.<作业号>.out` 和 `.err`。第 4、5 步中的直接运行命令仅适用于已经进入 GPU 节点的终端。

## 4. 单题试跑

**做什么：**仅对第 0 题生成 3 个回答，验证模型加载和图片输入。此步不调用判分 API。

```bash
SIUO_START=0 SIUO_END=0 bash scripts/run_siuo_generate.sh
```

预期产生 `outputs/test_results/siuo-gen-qwen25vl7b-grpo-step529.json`。该 JSON 包含全部 167 道题，但此时只有第 0 题应有 3 个回答。`SIUO_START`、`SIUO_END` 从 0 开始计数，且结束编号包含在范围内。

## 5. 全量推理

**做什么：**对剩余题目生成回答；脚本会跳过已完成的第 0 题，并在每次生成后保存进度。此步仍不调用判分 API。

```bash
bash scripts/run_siuo_generate.sh
```

推理完成后检查 167 道题是否各有 3 个回答：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/test_results/siuo-gen-qwen25vl7b-grpo-step529.json")
items = json.loads(path.read_text(encoding="utf-8"))
complete = sum(len(item.get("responses", [])) == 3 for item in items)
print(f"completed questions: {complete}/{len(items)}; responses: {sum(len(item.get('responses', [])) for item in items)}")
assert len(items) == complete == 167
PY
```

预期为 `167/167` 道完成、501 个回答。

## 6. 检查判分服务

**做什么：**在 `eval/gpt-eval.py` 的 `JUDGE_MODEL_CONFIGS` 中，为要使用的判分模型填写有效的 `api_key` 和 `base_url`。提交包中的值均为占位符，未填写时程序会直接报错。默认判分模型为 `gpt-5.5`。正常情况下需判分 501 个回答；失败重试会增加 API 请求数，可能产生费用。

只有模型推理结果时，可停在第 5 步；判分需要有效的 API 密钥及服务连接。

## 7. GPT 判分

**做什么：**读取已有的 501 个回答，调用 API 评估安全性，并检查每道题是否得到 3 个有效分数。此阶段不会加载生成模型，也不会计算汇总分数。运行前需等待第 5 步完成。

```bash
python eval/gpt-eval.py
```

预期成功信息包含 `GPT evaluation complete`。判分过程可中断后重跑，已完成的题目会被复用。如果判分结果不完整，程序会报错。试跑前 5 题可加 `--start 0 --end 4`；切换已配置的判分服务可加 `--judge-model`。

## 8. 汇总分数

**做什么：**读取已有判分文件，检查分数是否完整，并计算整体与分类安全率。此阶段不调用 API。

```bash
python eval/gen_score.py
```

程序要求全部 167 道题判分完成，随后自动写入 `final_score.json` 并在终端打印结果。

## 9. 查看结果

**做什么：**查看原始回答、逐题判分和最终汇总。

```bash
cat outputs/gpt_eval/qwen25vl7b-grpo-step529/gpt-5.5/final_score.json
```

- 模型回答：`outputs/test_results/siuo-gen-qwen25vl7b-grpo-step529.json`
- 逐题分数：`outputs/gpt_eval/qwen25vl7b-grpo-step529/<JUDGE_MODEL>/siuo_gen-qwen25vl7b-grpo-step529-gpteval-safe-<JUDGE_MODEL>-pro.json`
- 最终汇总：`outputs/gpt_eval/qwen25vl7b-grpo-step529/<JUDGE_MODEL>/final_score.json`

`final_score.json` 中的 `overall_avg_safety` 是总体安全率（百分数），`category_averages` 是各类别安全率。若更换模型文件，应同时设置新的 `MODEL_DIR` 和唯一的 `MODEL_NAME`，避免复用另一模型的旧回答。
