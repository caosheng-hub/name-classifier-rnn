# -*- coding:utf-8 -*-
import torch
import json
import os
import string
from torch.utils.data import DataLoader
# 复用原代码中的核心类和函数
from name_classifier_rnn import *

# ===================== 测试配置 =====================
TEST_NAME = "test_name.txt"  # 自定义测试人名文件
MODEL_DIR = "./save_model"  # 模型保存目录
PREDICT_NAMES = ["Zhang", "Smith", "Garcia", "Kim", "Rossi", "Lee"]  # 测试人名列表
BATCH_SIZE = 1
DEVICE = torch.device("cpu")  # 若有GPU可改为 "cuda"


# ===================== 工具函数 =====================
def create_test_data_file():
    """创建测试用的人名-国家映射文件（模拟真实数据格式）"""
    test_data = [
        "Zhang\tChinese",
        "Li\tChinese",
        "Wang\tChinese",
        "Smith\tEnglish",
        "Johnson\tEnglish",
        "Garcia\tSpanish",
        "Rodriguez\tSpanish",
        "Kim\tKorean",
        "Park\tKorean",
        "Rossi\tItalian",
        "Ferrari\tItalian",
        "Lee\tKorean",
        "Martin\tFrench",
        "Dubois\tFrench"
    ]
    with open(TEST_NAME, "w", encoding="utf-8") as f:
        f.write("\n".join(test_data))
    print(f"✅ 测试数据文件已创建：{TEST_NAME}")


def check_model_files():
    """检查训练好的模型文件是否存在"""
    required_models = ["ai23_rnn_1.bin", "ai23_lstm_1.bin", "ai23_gru_1.bin"]
    missing = []
    for model in required_models:
        if not os.path.exists(os.path.join(MODEL_DIR, model)):
            missing.append(model)
    if missing:
        raise FileNotFoundError(
            f"❌ 缺失模型文件：{missing}\n请先运行原代码的 train_rnn()/train_lstm()/train_gru() 训练模型")
    print("✅ 所有模型文件已找到")


def load_model(model_type):
    """加载指定类型的模型（RNN/LSTM/GRU）"""
    input_size = n_letters
    hidden_size = 128
    output_size = len(categorys)

    if model_type == "RNN":
        model = NameRNN(input_size, hidden_size, output_size).to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "ai23_rnn_1.bin")
    elif model_type == "LSTM":
        model = NameLSTM(input_size, hidden_size, output_size).to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "ai23_lstm_1.bin")
    elif model_type == "GRU":
        model = NameGRU(input_size, hidden_size, output_size).to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "ai23_gru_1.bin")
    else:
        raise ValueError(f"❌ 不支持的模型类型：{model_type}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # 切换到评估模式
    print(f"✅ {model_type} 模型加载完成")
    return model


# ===================== 核心测试函数 =====================
def test_data_processing():
    """测试数据读取和Dataset/DataLoader处理流程"""
    print("\n=== 测试1：数据处理流程 ===")
    # 1. 读取测试数据
    my_list_x, my_list_y = read_data(TEST_NAME)
    assert len(my_list_x) == len(my_list_y), "❌ 人名和国家数量不匹配"
    print(f"✅ 读取测试数据：共{len(my_list_x)}条样本")

    # 2. 测试Dataset
    dataset = NameDataset(my_list_x, my_list_y)
    assert len(dataset) == len(my_list_x), "❌ Dataset长度错误"
    x_tensor, y_tensor = dataset[0]
    assert x_tensor.shape == (len(my_list_x[0]), n_letters), "❌ 人名张量维度错误"
    assert y_tensor.dtype == torch.long, "❌ 标签类型错误"
    print(f"✅ Dataset测试通过：第一条样本 - 人名{my_list_x[0]} → 国家{categorys[y_tensor.item()]}")

    # 3. 测试DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    batch_x, batch_y = next(iter(dataloader))
    assert batch_x[0].shape == x_tensor.shape, "❌ DataLoader输出维度错误"
    print("✅ DataLoader测试通过")


def test_name2tensor():
    """测试人名转one-hot张量函数"""
    print("\n=== 测试2：人名转张量 ===")
    test_name = "test"
    tensor = name2tensor(test_name)
    assert tensor.shape == (len(test_name), n_letters), "❌ 张量维度错误"
    # 验证one-hot编码正确性
    for idx, letter in enumerate(test_name):
        assert tensor[idx][string.ascii_letters.find(letter)] == 1.0, f"❌ {letter}编码错误"
    print(f"✅ 人名'{test_name}'转张量测试通过，维度：{tensor.shape}")


def test_single_predict(model_type, name):
    """测试单个人名的预测功能"""
    model = load_model(model_type)
    tensor_x = name2tensor(name).to(DEVICE)

    with torch.no_grad():
        if model_type == "RNN" or model_type == "GRU":
            h0 = model.init_hidden().to(DEVICE)
            output, _ = model(tensor_x, h0)
        elif model_type == "LSTM":
            h0, c0 = model.init_hidden()
            h0, c0 = h0.to(DEVICE), c0.to(DEVICE)
            output, _, _ = model(tensor_x, h0, c0)

        # 取Top3预测结果
        topv, topi = torch.topk(output, k=3, dim=1)
        results = []
        for i in range(3):
            prob = torch.exp(topv[0][i]).item()  # LogSoftmax还原为概率
            country = categorys[topi[0][i].item()]
            results.append((country, round(prob, 3)))
    return results


def test_batch_predict():
    """批量测试不同模型的预测结果"""
    print("\n=== 测试3：批量人名预测 ===")
    for model_type in ["RNN", "LSTM", "GRU"]:
        print(f"\n----- {model_type} 预测结果 -----")
        for name in PREDICT_NAMES:
            results = test_single_predict(model_type, name)
            print(f"人名：{name:8s} → 预测结果：{results}")


def test_model_accuracy():
    """测试模型在测试集上的准确率"""
    print("\n=== 测试4：模型准确率评估 ===")
    # 加载测试数据
    my_list_x, my_list_y = read_data(TEST_NAME)
    dataset = NameDataset(my_list_x, my_list_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for model_type in ["RNN", "LSTM", "GRU"]:
        model = load_model(model_type)
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x[0].to(DEVICE)  # 去除batch维度
                y = y.to(DEVICE)

                if model_type == "RNN" or model_type == "GRU":
                    h0 = model.init_hidden().to(DEVICE)
                    output, _ = model(x, h0)
                elif model_type == "LSTM":
                    h0, c0 = model.init_hidden()
                    h0, c0 = h0.to(DEVICE), c0.to(DEVICE)
                    output, _, _ = model(x, h0, c0)

                pred = torch.argmax(output, dim=1)
                correct += (pred == y).sum().item()
                total += 1

        accuracy = correct / total * 100
        print(f"{model_type} 准确率：{correct}/{total} = {accuracy:.2f}%")


def test_result_files():
    """测试训练结果JSON文件的完整性"""
    print("\n=== 测试5：训练结果文件验证 ===")
    required_files = ["rnn_result.json", "lstm_result.json", "gru_result.json"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ 缺失结果文件：{file}")

        with open(file, "r") as f:
            data = json.load(f)
        # 验证关键字段
        assert "total_loss_list" in data, f"❌ {file} 缺失total_loss_list字段"
        assert "all_time" in data, f"❌ {file} 缺失all_time字段"
        assert "total_acc_list" in data, f"❌ {file} 缺失total_acc_list字段"
        print(f"✅ {file} 验证通过，损失记录数：{len(data['total_loss_list'])}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        # 1. 准备测试数据
        if not os.path.exists(TEST_NAME):
            create_test_data_file()

        # 2. 检查模型文件
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        check_model_files()

        # 3. 执行各项测试
        test_data_processing()
        test_name2tensor()
        test_batch_predict()
        test_model_accuracy()
        test_result_files()

        print("\n🎉 所有测试用例执行完成！")

    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
    finally:
        # 清理测试文件（可选）
        if os.path.exists(TEST_NAME):
            os.remove(TEST_NAME)
            print(f"\n🗑️ 测试文件已清理：{TEST_NAME}")