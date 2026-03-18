
---

```markdown id="resume_pro_readme"
# 🚀 NLP Sequence Modeling Benchmark  
### RNN vs LSTM vs GRU （人名分类任务）

> ⭐ 一个从 0 到 1 手写实现多种序列模型，并进行系统性对比实验的 NLP 项目  
> 👉 重点体现：**模型理解 + 工程能力 + 实验分析能力**

---

## 🧠 项目背景

在自然语言处理任务中，序列建模是核心问题之一。

本项目围绕一个经典任务展开：

> 🎯 **人名分类任务（Name Classification）**  
> 输入一个人名 → 预测其所属国家（18分类）

通过该任务，系统对比以下模型：

- RNN（基础模型）
- LSTM（长依赖优化）
- GRU（轻量高效）

---

## 🎯 项目亮点（面试重点）

✅ 从零实现 RNN / LSTM / GRU（非调用API）  
✅ 模块化实现（Encoder / Decoder 分离）  
✅ 完整训练流程（Dataset / DataLoader / Train Loop）  
✅ 多模型统一实验框架（可复用）  
✅ 系统对比：**收敛速度 / 精度 / 训练耗时**  
✅ 实验结果可视化（Loss / Accuracy / Time）

---

## 🏗️ 项目结构

```

.
├── RNN案例——人名分类器.py     # 模型定义
├── demo.py                   # 训练入口（统一控制实验）
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── name_classfication.txt
├── ai23_avg_acc.png
├── ai23_avg_loss.png
├── ai23_time.png

````

---

## ⚙️ 技术方案

### 🔹 数据处理

- 字符级输入（Character-level）
- One-hot 编码
- 使用 `Dataset + DataLoader` 构建数据管道

---

### 🔹 模型设计

#### 1️⃣ RNN

```text
h_t = tanh(Wx_t + Uh_{t-1})
````

* 优点：结构简单
* 缺点：梯度消失

---

#### 2️⃣ LSTM

引入三大门控机制：

* 输入门
* 遗忘门
* 输出门

👉 解决长期依赖问题

---

#### 3️⃣ GRU

* 合并门结构（更新门 + 重置门）
* 参数更少，训练更快

---

## 🧪 实验设计

统一实验配置：

* 相同数据集
* 相同训练轮数
* 相同优化器

对比维度：

| 指标       | 说明    |
| -------- | ----- |
| Loss     | 收敛情况  |
| Accuracy | 分类准确率 |
| Time     | 训练耗时  |

---

## 📊 实验结果

### 🔹 Loss 对比

![Loss](./assets/ai23_avg_loss.png)

👉 结论：

* RNN 收敛最慢
* LSTM / GRU 更稳定

---

### 🔹 Accuracy 对比

![Accuracy](./assets/ai23_avg_acc.png)

👉 结论：

* GRU 表现最佳
* LSTM 次之
* RNN 最差

---

### 🔹 训练时间

![Time](./assets/ai23_time.png)

👉 结论：

* GRU 速度最快

---

## 📌 实验结论（面试可直接说）

> 在该任务中：

* ✅ **GRU 在性能和效率之间取得最佳平衡**
* ✅ LSTM 精度略高但训练成本更大
* ❌ RNN 存在明显梯度问题

---

## 🧩 核心代码设计（亮点）

### 🔹 数据管道

```python
class NameDataset(Dataset):
```

* 支持动态序列长度
* One-hot 编码

---

### 🔹 模型统一接口

```python
model = RNN(...)
model = LSTM(...)
model = GRU(...)
```

👉 实现**可插拔模型设计**

---

### 🔹 训练流程

```python
for epoch in range(EPOCHS):
    for x, y in dataloader:
        ...
```

* 支持 loss 记录
* 支持结果导出 JSON

---

## 🚀 快速开始

```bash
pip install -r requirements.txt
python demo.py
```

---

## 📈 结果文件

```text
rnn_result.json
lstm_result.json
gru_result.json
```

记录：

* loss 曲线
* acc 曲线
* 训练过程

---
