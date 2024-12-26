import torch
import torch.nn as nn
import torch.nn.functional as F


# NPL 随机选取一段话, 然后移动这段话的位置, 使话与话之间的关系变得更加复杂
# x : 289690 289691 289692 289693
# y :        289691 289692 289693 289694
def get_batch(data, i, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# 定义一个简单的语言模型
class BLM(nn.Module):
    # 构造函数,初始化模型
    def __init__(self, vocab_size):
        super().__init__()
        # 定义一个词汇嵌入表, 每个词汇的嵌入向量的维度是词汇表大小
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # 前向传播函数, 接受输入的词索引 ( idx ) 和目标 ( targets )
    def forward(self, idx, targets=None):
        # 使用词汇嵌入表获取词向量的输出（logits）,即为输入的词索引的嵌入表示
        logits = self.token_embedding_table(idx)

        # 如果没有提供目标标签,则不计算损失
        if targets is None:
            loss = None
        else:
            # 获取 logits 的形状,B 是批次大小,T 是序列长度,C 是词汇大小
            B, T, C = logits.shape
            # 将logits展平为一个二维矩阵,形状变为（B * T, C）
            logits = logits.view(B * T, C)
            # 将目标标签展平为一维,形状变为 ( B * T )
            targets = targets.view(B * T)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, targets)

        # 返回logits和损失（如果有的话）
        return logits, loss


    # 反向传播函数, 返回生成的文本
    def generate(self, idx, max_new_tokens):
        """
        idx 是现在的输入的(B, T)序列 ,这是之前我们提取的batch的下标
        max_new_tokens 是产生的最大的tokens数量
        """
        for _ in range(max_new_tokens):
            # 得到预测的结果
            logits, _ = self(idx)  # _ 表示省略, 用于不获取相对应的函数返回值
            # 只关注最后一个的预测  (B,T,C)
            logits = logits[:, -1, :]  # becomes (B, C)
            # 对概率值应用 softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # nn.argmax
            # 对 input 的每一行做 n_samples 次取值, 输出的张量是每一次取值时input张量对应行的下标, 也即找到概率值输出最大的下标, 也对应着最大的编码
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将新产生的编码加入到之前的编码中,形成新的编码
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train(data, device, chars):
    # 划分训练集和验证集
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 模型参数设置
    vocab_size = len(chars)  # 字符集的大小
    # print("字符集的大小为 : " + str(vocab_size)) # 4006

    batch_size = 4  # 同时运行的批次大小
    block_size = 16  # 每个单元的最大长度

    max_iters = 1000  # 最大迭代次数
    learning_rate = 0.3  # 学习率 ( 深度学习中, 梯度下降的步长 )

    eval_iters = 200  # 每次评估模型的迭代次数
    eval_interval = 300  # 每隔多少次迭代评估一次模型

    # 模型初始化
    model = BLM(vocab_size).to(device)

    # 优化器 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 训练模型
    for i in range(max_iters):
        xb, yb = get_batch(train_data, 0, block_size, batch_size)
        logits, loss = model(xb, yb)    # 计算损失

if __name__ == "__main__":
    # 读取西游记全文
    with open("./data/Chinese/The Journey to the West.txt", "r", encoding="utf-8") as f:
        text = f.read()
    # print("西游记全文的长度为 : " + str(len(text)) + '\n' + "西游记的类型是 : " + str(type(text)))

    # 生成字符集 ( 文本中出现的所有字符 )
    # chars = sorted(list(set(text)))
    # print(len(chars))

    # 选取训练集
    n = int(0.5 * len(text))  # 取 50% 的数量
    text = text[:n]  # 重新映射文本
    chars = sorted(list(set(text)))  # 重新映射字符集

    # 对字符进行编码
    stoi = {ch: i for i, ch in enumerate(chars)}  # 创建一个字符到整数的哈希表
    itos = {i: ch for i, ch in enumerate(chars)}  # 创建一个整数到字符的哈希表

    encode = lambda s: [stoi[c] for c in s]  # 这是编码器, 用于将编码器编码成整数的向量
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # 这是解码器, 用于将整数向量解码回原来的文字

    # print(encode("西游记"))         # 测试编码器
    # print(decode(encode("西游记"))) # 测试解码器

    # 将文本编码成整数向量, 指定设备, 用于训练
    data = encode(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(data, device, chars)
