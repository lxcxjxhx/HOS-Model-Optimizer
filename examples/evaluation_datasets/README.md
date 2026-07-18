# 网络安全评测数据集

本目录包含用于评估大语言模型在网络安全领域能力的示例评测数据集。

## 数据集概述

提供两种标准格式的评测数据集：

- **Alpaca 格式**：`cybersecurity_eval_alpaca.json`
- **ShareGPT 格式**：`cybersecurity_eval_sharegpt.json`

两个数据集都包含 10-15 条高质量的网络安全领域问答样本，覆盖常见的安全主题。

## 数据格式说明

### Alpaca 格式

Alpaca 格式适用于单轮指令-响应评测，每条数据包含以下字段：

```json
{
  "instruction": "指令或问题",
  "input": "可选的输入上下文",
  "output": "期望的模型输出"
}
```

**示例：**
```json
{
  "instruction": "解释SQL注入攻击的原理，并给出一个防御示例。",
  "input": "",
  "output": "SQL注入是一种通过将恶意SQL代码插入应用程序查询中来攻击数据库的攻击方式..."
}
```

### ShareGPT 格式

ShareGPT 格式适用于多轮对话评测，每条数据包含一个对话列表：

```json
{
  "conversations": [
    {"from": "human", "value": "用户的问题"},
    {"from": "gpt", "value": "模型的回复"},
    {"from": "human", "value": "用户的追问"},
    {"from": "gpt", "value": "模型的进一步回复"}
  ]
}
```

**示例：**
```json
{
  "conversations": [
    {"from": "human", "value": "请帮我分析这段代码是否存在安全漏洞..."},
    {"from": "gpt", "value": "这段代码存在典型的SQL注入漏洞..."},
    {"from": "human", "value": "如果除了id还有其他字符串类型的参数，应该怎么处理？"},
    {"from": "gpt", "value": "对于字符串类型参数，同样使用PreparedStatement..."}
  ]
}
```

## 覆盖主题

数据集覆盖以下网络安全核心主题：

1. **Web应用安全**
   - SQL注入攻击与防御
   - 跨站脚本攻击（XSS）
   - 跨站请求伪造（CSRF）
   - 路径遍历漏洞

2. **身份认证与授权**
   - 多因素认证（MFA）
   - OAuth 2.0 协议
   - JWT 安全风险
   - Session 管理

3. **加密与数据保护**
   - 对称加密与非对称加密
   - TLS/SSL 配置
   - 密码哈希与存储
   - HSTS 与证书管理

4. **渗透测试**
   - 渗透测试标准流程
   - 社会工程学攻击
   - 内网渗透与域控攻击
   - 漏洞利用技术

5. **网络安全监控**
   - 网络日志分析
   - Wireshark 流量分析
   - 异常行为检测
   - 端口扫描识别

6. **安全架构与最佳实践**
   - 零信任架构（Zero Trust）
   - 容器安全
   - DDoS 攻击与防御
   - 勒索软件防护
   - 零日漏洞应对

## 使用方法

### 加载数据集

```python
import json

# 加载 Alpaca 格式
with open('cybersecurity_eval_alpaca.json', 'r', encoding='utf-8') as f:
    alpaca_data = json.load(f)

# 加载 ShareGPT 格式
with open('cybersecurity_eval_sharegpt.json', 'r', encoding='utf-8') as f:
    sharegpt_data = json.load(f)

print(f"Alpaca 数据集包含 {len(alpaca_data)} 条样本")
print(f"ShareGPT 数据集包含 {len(sharegpt_data)} 条样本")
```

### 在评测中使用

```python
# Alpaca 格式评测示例
for item in alpaca_data:
    instruction = item['instruction']
    input_context = item.get('input', '')
    expected_output = item['output']
    
    # 构建提示
    prompt = f"指令：{instruction}\n"
    if input_context:
        prompt += f"输入：{input_context}\n"
    prompt += "回答："
    
    # 调用模型生成回复
    model_output = model.generate(prompt)
    
    # 评估模型输出（可使用 BLEU、ROUGE 或人工评估）
    score = evaluate(model_output, expected_output)
    print(f"得分：{score}")

# ShareGPT 格式评测示例
for conversation_data in sharegpt_data:
    messages = conversation_data['conversations']
    
    # 构建多轮对话历史
    history = []
    for msg in messages:
        if msg['from'] == 'human':
            history.append({"role": "user", "content": msg['value']})
        elif msg['from'] == 'gpt':
            history.append({"role": "assistant", "content": msg['value']})
    
    # 逐轮评测
    for i, msg in enumerate(messages):
        if msg['from'] == 'gpt':
            # 构建到当前轮次的对话历史
            eval_history = history[:i]
            expected = msg['value']
            
            # 调用模型
            model_output = model.chat(eval_history)
            
            # 评估
            score = evaluate(model_output, expected)
            print(f"第 {i//2 + 1} 轮对话得分：{score}")
```

### 使用 Hugging Face datasets 库

```python
from datasets import load_dataset

# 加载本地 JSON 文件
alpaca_dataset = load_dataset('json', data_files='cybersecurity_eval_alpaca.json')
sharegpt_dataset = load_dataset('json', data_files='cybersecurity_eval_sharegpt.json')

# 查看数据集结构
print(alpaca_dataset)
print(sharegpt_dataset)

# 访问样本
sample = alpaca_dataset['train'][0]
print(sample['instruction'])
print(sample['output'])
```

## 评测指标建议

针对网络安全领域的评测，建议使用以下指标：

1. **准确性**：技术概念解释是否正确
2. **完整性**：是否覆盖了问题的关键方面
3. **实用性**：提供的解决方案是否可操作
4. **安全性**：是否遵循安全最佳实践，不提供有害指导
5. **代码质量**：代码示例是否安全、规范

可以使用：
- **自动评估**：BLEU、ROUGE、BERTScore
- **人工评估**：由网络安全专家进行评分
- **混合评估**：自动指标初筛 + 人工复核

## 注意事项

1. **数据质量**：本数据集为示例数据，实际评测时建议根据具体需求扩充样本数量和多样性
2. **领域适配**：数据集主要针对中文网络安全场景，如需其他语言或特定子领域，请自行扩展
3. **伦理合规**：数据集中的渗透测试和攻击技术仅用于防御和教育目的，请遵守相关法律法规
4. **版本更新**：网络安全领域快速发展，建议定期更新数据集以反映最新威胁和技术

## 扩展建议

如需扩展数据集，可以考虑：

1. 增加更多网络安全子主题（如云安全、移动安全、IoT安全）
2. 添加代码审计类样本（提供代码片段，要求识别漏洞）
3. 增加场景分析类样本（提供日志或配置文件，要求分析问题）
4. 引入多语言支持（英文、日文等）
5. 增加难度分级（初级、中级、高级）

## 许可证

本示例数据集遵循与主项目相同的许可证。

## 贡献

欢迎提交 Pull Request 来改进和扩展本数据集。提交前请确保：

1. 数据格式符合 Alpaca 或 ShareGPT 规范
2. 内容准确、专业，经过验证
3. 覆盖新的主题或补充现有主题的样本
4. JSON 格式正确，可以被正常解析
