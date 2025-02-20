# RolePlay-LLM
基于llama-factory框架训练的角色扮演模型，数据参考https://github.com/LC1332/Chat-Haruhi-Suzumiya/tree/main

## 训练步骤
1. 先下载数据集
   https://huggingface.co/datasets/silk-road/ChatHaruhi-Expand-118K
2. 安装llama-factory框架，拉去git代码库，下载基座模型
3. 目前只用了Haruhi54K.jsonl这份数据集，运行prepare_data.py提取出多轮对话形式的训练数据（数据涉及多个人物，而非常规的一问一答，这也是从角色原始作品中提取数据常见的情况）
    - 每条数据拆分成多个sceen，每个sceen根据说话者拆成多轮对话，非角色保留说话者人名，角色仅保留内容
    - 保留由role发起的对话内容，尝试训练具有自主聊天的机器人
    - 数据去重复：343480 -> 64241
4. 得到的数据记得去重复，最后数量是64241个多轮对话
5. llama_factory/data/dataset_info.json添加数据集信息
     "chatharuhi54k-train":{
    "file_name": "$具体地址",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
6. llama_factory/src/llamafactory/data/template.py添加模板内容
  _register_template(
    name="qwen-roleplay",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="",
    stop_words=["<|im_end|>"],
)
7. 开始训练
FORCE_TORCHRUN=1 llamafactory-cli train /root/autodl-tmp/train_config/train.yaml
## 注意事项
0.5b的基座可：单卡4090（D）24G，per-device-batch=4
1.5b的基座可：双卡4090(D)24G, per-device-batch=1
