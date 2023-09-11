import sys
sys.dont_write_bytecode = True
import argparse
import os
import nni
import time
import pickle
import random
import setproctitle
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from transformers import (AdamW,AlbertTokenizer, AlbertModel,
                          get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, BertTokenizer)

from src.config import get_argparse
from src.metrics import flat_accuracy, flat_f1
from model import NewBert
from logger import logger
from src.utils import convert_examples_to_features, read_examples
# from CrossModel import Cross_Model # 导入cross_attention模型
from nni.utils import merge_parameter

args = get_argparse().parse_args()
args = vars(args)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 禁止并行
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
procname = str(args["name"]) + "test" if args["test"] else str(args["name"])
setproctitle.setproctitle('wyh_{}'.format(procname))
tasks = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE"]

# 定义gpu设备
device = torch.cuda.current_device()
args["device"] = device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

class MyDataset(Dataset):
    def __init__(self, features, features_chunks, labels):
        self.features = features
        self.features_chunks = features_chunks
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.features_chunks[index], self.labels[index]

    def __len__(self):
        return len(self.features)

def get_dataloader(args, features, features_chunks, labels, batch_size,  is_training):
    '''加载数据构建dataset，获取dataloader'''
    dataset = MyDataset(features=features, features_chunks=features_chunks, labels=labels)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=is_training,
                            num_workers=4)
    return dataloader

def save_model(args, model):
    time_now = int(time.time())
    #转换成localtime
    time_local = time.localtime(time_now)
    #转换成新的时间格式(2016-05-09 18:59:20)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    
    if not os.path.exists(args["model_dir"]):
        os.makedirs(args["model_dir"])
    ckpt_file = os.path.join(args.model_dir, "bert_base_{}.pt".format(args["name"]))
    torch.save(model, ckpt_file)

def test(args, model, device, tokenizer, albert_tokenizer):
    if args["read_data"]:
        examples, chunks = read_examples(args["dev_file"], args["name"],
                                        is_training=False)

        # chunks_examples是直接返回的tokenizer之后的内容；
        test_features, test_chunks_features, labels = convert_examples_to_features(args, examples, chunks, albert_tokenizer,tokenizer,
                                                args.max_len,
                                                is_training=False)
        pickle_file = os.path.join(args["pickle_folder"], "test_features_{}.pkl".format(args["name"]))
        test_data = {
            "test_features": test_features,
            "test_chunks_features": test_chunks_features,
            "labels":labels
        }
        with open(pickle_file, "wb") as f:
            pickle.dump(test_data, f)
            print("save {} test pickle file at: {}".format(args["name"], pickle_file))
    else:
        pickle_file = os.path.join(args["pickle_folder"], "test_features_{}.pkl".format(args["name"]))
        with open(pickle_file, "rb") as f:
            test_data = pickle.load(f)
        test_features, test_chunks_features, labels = test_data["test_features"], test_data["test_chunks_features"], test_data["labels"]
        
    test_dataloader = get_dataloader(args, test_features, test_chunks_features, labels, args["test_batch_size"], is_training=False)
    
    model.eval()
    
    result = []
    for i, (feature_batch, feature_chunk_batch, label_batch) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            # inputs_sentence, inputs_chunk, labels = process_batch(batch, device)
            feature_batch, feature_chunk_batch, label_batch = feature_batch.to(device), feature_chunk_batch.to(device), label_batch.to(device)
            loss, logits = model(feature_batch, feature_chunk_batch, label_batch)
            if args["baseline"] == 1:# 不同模型forward值不同
                loss, logits = loss[0], loss[1]
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            result.extend(pred_flat)
    indexs = [i for i in range(len(result))]
    # 处理部分任务的二分类
    if args["name"] in ["RTE", "QNLI"]:
        result = ["entailment" if tag else "not_entailment" for tag in result]
    
    # 处理MNLI三分类任务
    if "MNLI" in args["name"]:
        result_ = []
        for tag in result:
            if tag == 0:
                result_.append("contradiction")
            elif tag == 1:
                result_.append("neutral")
            else:
                result_.append("entailment")
        result = result_
                
    res = {
        "index":indexs,
        "prediction":result
    }   
    df = pd.DataFrame(res)
    df.to_csv('/data/wyh/graduate/data/submit/{}.tsv'.format(args["name"]), sep = '\t', index=False, header=True)
    
        
        
def evaluate(args, model, device, tokenizer, albert_tokenizer):
    if args["read_data"] or args["name"] == "MNLIMM":
        examples, chunks = read_examples(args["dev_file"], args["name"],
                                    is_training=False)
        eval_features, eval_chunks_features, labels = convert_examples_to_features(args, examples, chunks, albert_tokenizer,tokenizer,
                                                args["max_len"],
                                                is_training=False)
        if not os.path.exists(args["pickle_folder"]):
            os.makedirs(args["pickle_folder"])
        pickle_file = os.path.join(args["pickle_folder"], "evaluate_features_{}.pkl".format(args["name"]))
        eval_data = {
            "eval_features": eval_features,
            "eval_chunks_features": eval_chunks_features,
            "labels":labels
        }
        with open(pickle_file, "wb") as f:
            pickle.dump(eval_data, f)
            print("save pickle file at: {}".format(pickle_file))
    else:
        pickle_file = os.path.join(args["pickle_folder"], "evaluate_features_{}.pkl".format(args["name"]))
        with open(pickle_file, "rb") as f:
            eval_data = pickle.load(f)
        eval_features, eval_chunks_features, labels = eval_data["eval_features"], eval_data["eval_chunks_features"], eval_data["labels"]

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    eval_dataloader = get_dataloader(args, eval_features, eval_chunks_features, labels, args["test_batch_size"], is_training=False)
    
    model.eval()
    
    total_val_loss, total_eval_accuracy, total_eval_f1 = 0, 0, 0.0
    result = []
    for i, (inputs_sentence, inputs_chunk, labels) in enumerate(eval_dataloader):
        with torch.no_grad():
            inputs_sentence, labels = inputs_sentence.to(device), labels.to(device)
            loss, kl, logits = model(inputs_sentence["input_ids"], inputs_sentence["attention_mask"], inputs_sentence["token_type_ids"], None, labels)
            total_val_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            # 针对MRPC、QQP任务需要报告f1分数单独处理
            if args["name"] in ["MRPC", "QQP"]:
                total_eval_f1 += flat_f1(logits, label_ids)
            
            # 针对MRPC任务单独处理，因为其只有训练集和测试集
            if args["name"] == "MRPC":
                pred_flat = np.argmax(logits, axis=1).flatten()
                result.extend(pred_flat)
    # 针对MRPC任务单独处理，因为其只有训练集和测试集     
    if args["name"] == "MRPC":
        indexs = [i for i in range(len(result))]
        res = {
            "index":indexs,
            "prediction":result
        }   
        df = pd.DataFrame(res)
        df.to_csv('/data/wyh/graduate/data/submit/{}.tsv'.format(args["name"]), sep='\t', index=False, header=True)
    
    avg_val_loss = total_val_loss / len(eval_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)

    logger.info(f'Validation loss: {avg_val_loss}')
    logger.info(f'Accuracy: {avg_val_accuracy:.4f}')
    results = {
        "val_loss":avg_val_loss,
        "acc":avg_val_accuracy
    }
    if args["name"] == "MRPC":
        avg_val_f1 = total_eval_f1 / len(eval_dataloader)
        logger.info(f'F1: {avg_val_f1:.4f}')
        results = {
            "val_loss":avg_val_loss,
            "acc":avg_val_accuracy,
            "f1":avg_val_f1
        }
    return results

def run(args): 
    set_seed(args["seed"])
    tokenizer = BertTokenizer.from_pretrained(args["model"])
    albert_tokenizer = BertTokenizer.from_pretrained(args["model"])
    # 根据任务修改文件名
    TASKS = ["SST-B", "MRPC", "QQP", "MNLI", "QNLI", "RTE"]
    # root = "/data/wyh/graduate/data/glue"
    root = "/data/wyh/graduate/AugData"
    if args["name"] == "STS-B":args["n_class"] = 5
    elif args["name"] == "SICK":args["n_class"] = 3
    
    args["train_file"] = os.path.join(root, args["name"])
    args["dev_file"] = os.path.join(root, args["name"])
    

    name = "MNLI" if "MNLI" in args["name"] else args["name"]
    # 加载模型&测试数据写入pickle
    if args["test"]:
         # load the best model from validation-set
        ckpt_file = os.path.join(args["model_dir"], "bert_base_{}.pt".format(args["name"]))
        model = torch.load(ckpt_file, map_location="cpu")
        model = model.to(device)
        test(args, model, device, tokenizer, albert_tokenizer)
        logger.info("{}测试写入文件结束！".format(args["name"]))
        return
    
    # 训练集features写入到pkl
    if args["read_data"]:
        train_examples, train_chunks = read_examples(args["train_file"], args["name"],
                                             is_training=True)

        train_features, train_chunks_features, labels = convert_examples_to_features(args, train_examples, train_chunks, albert_tokenizer,
                                                      tokenizer,
                                                      args["max_len"],
                                                      is_training=True)
        if not os.path.exists(args["pickle_folder"]):
            os.makedirs(args["pickle_folder"])
        pickle_file = os.path.join(args["pickle_folder"], "train_features_{}.pkl".format(name))
        train_data = {
            "train_features": train_features,
            "train_chunks_features": train_chunks_features,
            "labels":labels
        }
        with open(pickle_file, "wb") as f:
            pickle.dump(train_data, f)
            print("save pickle file at: {}".format(pickle_file))

    else:
        pickle_file = os.path.join(args["pickle_folder"], "train_features_{}.pkl".format(name))
        assert os.path.exists(
            pickle_file) == True, "you must create pickle file set option --read_data"

        with open(pickle_file, "rb") as f:
            train_data = pickle.load(f)
        train_features, train_chunks_features, labels = train_data["train_features"], train_data["train_chunks_features"], train_data["labels"]
        if not args["test"]:
            length = int(args['ratio'] * len(train_data["train_features"]))
        else:
            length = len(train_data["train_features"])
        train_features, train_chunks_features, labels = train_features[:length], train_chunks_features[:length], labels[:length]
        # print(train_features)
    train_loader = get_dataloader(args, train_features, train_chunks_features, labels, args["batch_size"], is_training=True)
    
    # 模型
    encodermodel = NewBert(args) # 主模型
    if args["baseline"] or args["aug"]:model = encodermodel
    else:
        model = encodermodel
    # 加载checkpoint启用
    # ckpt_file = os.path.join(args.model_dir, "bert_base_{}.pt".format(args.name))
    # model = torch.load(ckpt_file, map_location="cpu")
    model = model.to(device)
    
    t_total = len(train_loader) * args["num_train_epochs"]

    optimizer = AdamW(model.parameters(),
                      lr=args["learning_rate"], eps=args["adam_epsilon"])

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total) * args["warm_up_rate"],
                                                num_training_steps=t_total)
    
    loss_log = tqdm(total=0, bar_format='{desc}', position=1)
    
    eval_acc, eval_loss = 0.0, 100
    
    for iter in range(args["num_train_epochs"]):
        model.train()
        num_batches = len(train_loader)
        if args["read_data"]:results = evaluate(args, model, device, albert_tokenizer, tokenizer)
        for index, (inputs_sentence, inputs_chunk, labels) in enumerate(tqdm(train_loader, total=num_batches, position=0, leave=False)):
            inputs_sentence, labels = inputs_sentence.to(device), labels.to(device)
            outputs = model(inputs_sentence["input_ids"], inputs_sentence["attention_mask"], inputs_sentence["token_type_ids"], None, labels)
            if args["baseline"]:outputs = outputs[0]
            if args["aug"]:
                    nll, kl, logits = outputs[0], outputs[1], outputs[2]
                    loss = nll + kl * args["beta"]
                    loss_str = "NLL: {:.4f}, KL: {:.4f}".format(
                        nll.item(), kl.item())
            else:
                loss = outputs[0]
                loss_str = "NLL: {:.4f}".format(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_log.set_description_str(loss_str)
            
        results = evaluate(args, model, device, albert_tokenizer, tokenizer)
        print("epoch:",iter)
    results = evaluate(args, model, device, tokenizer, albert_tokenizer)
    logger.info("best eval acc:{}".format(eval_acc))

if __name__ == "__main__":
    run(args)