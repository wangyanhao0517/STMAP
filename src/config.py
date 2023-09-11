'''
Descripttion: 
version: 
Author: wyh
Date: 2022-03-22 22:44:55
LastEditors: wyh
LastEditTime: 2022-03-30 14:32:13
'''
import warnings
import ast
from argparse import ArgumentParser

def get_argparse():
    parser = ArgumentParser(description="ADSM")

    # 超参数
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help = "训练 batch")
    parser.add_argument('--test_batch_size', type=int, default=256, help = "验证预测batch大小")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help = "学习率")
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument("--warm_up_rate", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--save_steps', type=int, default=5000, help = "每训练一定步数，就保存模型")
    parser.add_argument('--save_total_limit', type=int, default=10)

    parser.add_argument('--fp16', type=str, default=True, help = "模型中的数据精度，表示16浮点数")
    
    parser.add_argument("--n_class", type=int, default=2, help = "分类类别数量")

    parser.add_argument('--seed', type=int, default=1234, help = "随机种子")

    parser.add_argument('--uniform', type=int, default=0, help = "是否采用均匀分布的噪声")

    parser.add_argument('--hidden_size', type=int, default=768)
    
    parser.add_argument('--beta', type=float, default=0.5, help="aug使用的loss参数")
    
    parser.add_argument('--ratio', type=float, default=1, help="训练集合数据占比")

    parser.add_argument('--aug', type=int, default=1, help="是否使用字扰动增强方案")
    
    parser.add_argument('--test', type=int, default=0, help="是否为测试状态, 1表示测试，0表示训练或验证")
    
    parser.add_argument('--read_data', type=int, default=0, help="0表示读取pickle")
    
    parser.add_argument('--gate', type=int, default=1, help="0表示不使用gate网络")
    
    parser.add_argument('--baseline', type=int, default=0, help="0表示不使用baseline")
    # 几种不同的loss          
    parser.add_argument('--TruncatedTripletLoss', type=str, default=False, help = "TruncatedTripletLoss loss计算 \
                        需要一个list negative, 用于缓解过拟合和欠拟合 ")
    
    parser.add_argument('--device', type=str, default='cuda', help = "是否使用GPU")

    parser.add_argument('--dropout', type=float, default=0.1, help = "bert dropout的值")
    
    parser.add_argument('--zero_peturb', type=int, default=3, help = "bert dropout的值")

    # 多GPU设置

    parser.add_argument('--multiGPU', type=ast.literal_eval, default=False, help = "是否使用多GPU DDP策略进行训练")
    
    parser.add_argument('--nproc_per_node', type=int, default=2, help = "单机多卡，gpu数量")

    parser.add_argument("--local_rank", type=int, default=-1, help = "设置单机多卡使用的参数")
    
    parser.add_argument("--gpu_id", type=int, default=2, help = "运行当前代码使用的GPU卡编号")


    # 文本长度
    parser.add_argument('--output_dim', type=int, default=768, help = "最后进行cosin相似度计算的embeding纬度")

    parser.add_argument('--max_len', type=int, default=64, help = "pair长度")

    # 文件路径

    parser.add_argument('--test_file', type=str, default='/data/wyh/graduate/data/RTE/test.tsv')
    
    parser.add_argument('--dev_file', type=str, default='/data/wyh/graduate/data/RTE/dev.tsv')

    parser.add_argument('--train_file', type=str, default='/data/wyh/graduate/data/RTE/train.tsv')
    
    parser.add_argument('--output_dir', type=str, default='/data/wyh/graduate/data/output', \
        help = "验证结果、测试结果输出文件")
    
    parser.add_argument('--pickle_folder', type=str, default='/data/wyh/graduate/data/pickle', \
        help = "验证结果、测试结果输出文件")
    # 模型

    parser.add_argument('--model_dir', type=str, default='/data/wyh/graduate/data/save', \
        help = "模型保存的路径")
    
    parser.add_argument('--crossmodel', type=str, default='albert-base-v2', \
        help = "cross attention预训练模型的路径")
    
    parser.add_argument('--model', type=str, default='/data/wyh/graduate/data/bert-base-uncased', help = "预训练模型")

    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')

    # other
    parser.add_argument('--name', type=str, default='SICK', help="用来指明当前训练的进程名、tensorboard文件名、将要保存的模型名")
    
    return parser