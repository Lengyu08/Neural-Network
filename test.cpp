#include <cmath>
#include <fstream>
#include <iostream>
#include <typeinfo>

#include <cxxabi.h>

int main() {
    // 读取输入图片和模型参数
    std::ifstream input_file("./input.txt");
    std::ifstream p1_b_file("./p_1_b.txt");
    std::ifstream p1_w_file("./p_1_w.txt");
    std::ifstream p2_b_file("./p_2_b.txt");
    std::ifstream p2_w_file("./p_2_w.txt");

    float input[784];
    float p1_b[100], p1_w[78400], p2_b[10], p2_w[1000];
    for (int i = 0; i < 784; i++) input_file >> input[i];
    for (int i = 0; i < 100; i++) p1_b_file >> p1_b[i];
    for (int i = 0; i < 78400; i++) p1_w_file >> p1_w[i];
    for (int i = 0; i < 10; i++) p2_b_file >> p2_b[i];
    for (int i = 0; i < 1000; i++) p2_w_file >> p2_w[i];

    // 定义激活函数和预测函数
    auto tanh_fn = [](float x) -> float { return std::tanh(x); };
    auto softmax_fn = [](float x) -> float { return std::exp(x); };

    auto predict = [&](float input[784]) -> int {
        float l1[100];
        float l2[10];

        // 计算第一层结果
        for (int i = 0; i < 100; i++) {
            float sum = 0;
            for (int j = 0; j < 784; j++) {
                sum += input[j] * p1_w[j * 100 + i];
            }
            l1[i] = tanh_fn(sum + p1_b[i]);
        }

        // 计算第二层结果
        for (int i = 0; i < 10; i++) {
            float sum = 0;
            for (int j = 0; j < 100; j++) {
                sum += l1[j] * p2_w[j * 10 + i];
            }
            l2[i] = softmax_fn(sum + p2_b[i]);
        }

        // 找到概率最大的数字并返回
        int max_idx = 0;
        for (int i = 1; i < 10; i++) {
            if (l2[i] > l2[max_idx]) max_idx = i;
        }
        return max_idx;
    };
    // 输出预测结果
    std::cout << "Predicted number: " << predict(input) << std::endl;
    return 0;
}