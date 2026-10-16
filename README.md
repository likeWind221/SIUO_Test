# SIUO 测试交付包

本目录使用训练后的 Qwen2.5-VL GRPO step 529 模型，在 SIUO generation 数据集上每题生成 3 个回答，并可调用兼容 OpenAI 的 API 判分、汇总安全率。

## 环境

以下命令均在 `siuo_test_package/` 目录执行。参考环境为 Python 3.12.14、PyTorch 2.8.0+cu128；`requirements.txt` 已配置清华 PyPI 镜像。

```bash
conda create -n siuo-test python=3.12 -y \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate siuo-test
python -m pip install -r requirements.txt
python -m pip check
```

## 运行

**西电超算平台：**在登录节点提交 GPU 推理作业，脚本通过 `jsub` 申请 `gpu_6000ada` 队列的 1 张 GPU：

```bash
bash scripts/submit_siuo_generate.sh
```

等待作业完成并确认 `outputs/logs/` 中的推理日志后，再运行：

```bash
python eval/gpt-eval.py
python eval/gen_score.py
```

**其他服务器：**先进入已分配 GPU 的终端并激活 `siuo-test` 环境，再直接运行推理。推理结束后运行判分和汇总；若服务器使用其他调度器，应在 GPU 作业中执行推理命令。

```bash
bash scripts/run_siuo_generate.sh
python eval/gpt-eval.py
python eval/gen_score.py
```

判分模型、API 密钥和对应服务地址在 `eval/gpt-eval.py` 的 `JUDGE_MODEL_CONFIGS` 中配置。提交包只包含占位值；运行判分前需填写所用模型的密钥和地址。默认使用 `gpt-5.5`，可用 `--judge-model` 选择其他模型。两步 Python 程序不会加载生成模型，汇总程序也不会调用 API。

回答保存在 `outputs/test_results/siuo-gen-<MODEL_NAME>.json`；判分文件及 `final_score.json` 保存在 `outputs/gpt_eval/<MODEL_NAME>/<JUDGE_MODEL>/`。
