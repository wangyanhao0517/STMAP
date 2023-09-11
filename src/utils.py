import os
import sys
import pandas as pd
from copy import deepcopy
sys.path.append("/data/wyh/graduate/New/")
import json_lines
import numpy as np
from tqdm import tqdm
from transformers import BasicTokenizer
ROOT_DIR = os.path.abspath("/data/wyh/graduate/New")
sys.path.append(ROOT_DIR)
from block.calculate import PTM_keyword_extractor_yake
from src.config import get_argparse
args = get_argparse().parse_args()

class MatchExample(object):
    '''文本example'''
    def __init__(self, sentence1, sentence2, label) -> None:
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
class ChunkExample(object):
    def __init__(self, chunk1, chunk2, label) -> None:
        self.chunk1 = chunk1
        self.chunk2 = chunk2
        self.label = label
        
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label) -> None:
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

def read_RTE(input_file, is_training):
    '''二分类'''
    suffix = ["train.tsv", "dev.tsv"]
    if args.test:
        suffix = "test.tsv"
    else:suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, quoting=3)

    examples, chunks = [], []
    # 参考 https://blog.csdn.net/sinat_29675423/article/details/87972498
    for index, row in df.iterrows():
        if args.test == 1:
            label = 0
        else:
            if isinstance(row['label'], int):label = row["label"]
            else:
                label = 1 if row["label"] == "entailment" else 0
        example = MatchExample(row['sentence1'], row['sentence2'], label)
        # print(index, PTM_keyword_extractor_yake(row['sentence1']))
        print(index)
        chunk = ChunkExample(PTM_keyword_extractor_yake(row['sentence1']), PTM_keyword_extractor_yake(row['sentence2']), label)
        chunks.append(chunk)
        examples.append(example)
    return examples, chunks

def read_SICK_aug(input_file, is_training):
    '''二分类'''
    output = "/data/wyh/graduate/AugData/sick_backtrans.tsv"
    df = pd.read_csv(output, sep = '\t')
    examples, chunks = [], []
    index = 0
    for i, row in df.iterrows():
        print(index, row['sentence1'], row['sentence2'], row['label'])
        index += 1
        s1, s2, label = row['sentence1'], row['sentence2'], row['label']
        example = MatchExample(s1, s2, int(label))
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(s1), PTM_keyword_extractor_yake(s2), label)
        chunks.append(chunk)
    print("total train items:", index)
    return examples, chunks

def read_SICK(input_file, is_training):
    """
        测试集有label，直接将测试集当作验证集进行超参数搜索
    """
    examples, chunks = [], []
    if not is_training:
        suffix = ["train", "test"]
        suffix = suffix[0] if is_training else suffix[-1] 
        dir = '/data/wyh/graduate/NEW_SICK/SICK.txt'
        df = pd.read_csv(dir, sep = '\t', header=0, quoting=3)
        for index, row in df.iterrows():
            if (suffix == "test" and row["SemEval_set"] == "TEST") or (suffix == "train" and row["SemEval_set"] == "TRAIN"):
                # print(index)
                sentence1 = row["sentence_A"]
                sentence2 = row["sentence_B"]
                label = row["entailment_label"]
                if label == "ENTAILMENT":
                    label = 2
                elif label == "NEUTRAL":
                    label = 1
                else:
                    label = 0
                example = MatchExample(sentence1, sentence2, label)
                examples.append(example)
                chunks.append(0)
    else:
        dir = '/data/wyh/graduate/NEW_SICK/sick_new_train.tsv'
        df = pd.read_csv(dir, sep = '\t', header=0, quoting=3)
        for index, row in df.iterrows():
            print(index)
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            label = row["label"]
            example = MatchExample(sentence1, sentence2, label)
            examples.append(example)
            chunks.append(0)
    return examples, chunks
        
def read_QNLI(input_file, is_training):
    '''二分类'''
    suffix = ["train.tsv", "dev.tsv"]
    if args.test:
        suffix = "test.tsv"
    else:suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, on_bad_lines='skip', quoting=3)

    examples, chunks = [], []
    for index, row in df.iterrows():
        print(index)
        # if index > 10000:break
        if args.test:label = 0
        else:
            if isinstance(row['label'], int):label = row["label"]
            else:
                label = 1 if row["label"] == "entailment" else 0
        example = MatchExample(row['question'], row['sentence'], label)
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(row['question']), PTM_keyword_extractor_yake(row['sentence']), label)
        chunks.append(chunk)
    return examples, chunks

def read_QQP(input_file, is_training):
    '''二分类'''
    suffix = ["train.tsv", "dev.tsv"]
    if args.test:
        suffix = "test.tsv"
    else:
        suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, quoting=3)
    examples, chunks = [], []
    for index, row in df.iterrows():
        print(index)
        # if index > 1000:break
        label = 0 if args.test else int(row["is_duplicate"])
        example = MatchExample(row['question1'], row['question2'], label)
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(row['question1']), PTM_keyword_extractor_yake(row['question2']), label)
        chunks.append(chunk)
    return examples, chunks

def read_STSB(input_file, is_training):
    '''
    回归任务
    评价指标也不是acc 和 f1 最后考虑
    '''
    
    suffix = ["train.tsv", "dev.tsv"]
    suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, on_bad_lines='skip')

    examples, chunks = [], []
    for index, row in df.iterrows():
        print(index)
        example = MatchExample(row['sentence1'], row['sentence2'], int(row['score']))
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(row['sentence1']), PTM_keyword_extractor_yake(row['sentence2']), int(row['score']))
        chunks.append(chunk)
    return examples, chunks

def read_MRPC(input_file, is_training):
    '''二分类'''
    suffix = ["msr_paraphrase_train.txt", "msr_paraphrase_test.txt"]
    suffix = suffix[0] if is_training else suffix[-1] 
    # dir_ = os.path.join(input_file, suffix)
    dir_ = "/data/wyh/graduate/AugData/msr_paraphrase_test.txt"
    examples, chunks = [], []
    index = 0
    with open(dir_, encoding="utf-8") as f:
        for row in f:
            print(index)
            # if index > 200:break
            index += 1
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if len(label) > 1:continue
            example = MatchExample(s1, s2, int(label))
            examples.append(example)
            chunk = ChunkExample(PTM_keyword_extractor_yake(s1), PTM_keyword_extractor_yake(s2), label)
            chunks.append(chunk)
    return examples, chunks

def read_MRPC_aug(input_file, is_training):
    '''二分类'''
    output = "/data/wyh/graduate/AugData/mrpc_backtrans.tsv"
    df = pd.read_csv(output, sep = '\t')
    examples, chunks = [], []
    index = 0
    for i, row in df.iterrows():
        print(index, row['sentence1'], row['sentence2'], row['label'])
        index += 1
        s1, s2, label = row['sentence1'], row['sentence2'], row['label']
        example = MatchExample(str(s1), str(s2), int(label))
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(s1), PTM_keyword_extractor_yake(s2), label)
        chunks.append(chunk)
        # if i > 100:break
    print("total train items:", index)
    return examples, chunks

def read_MNLIM(input_file, is_training):
    '''0表示contradiction， 1表示neutral，2表示entailment;
        MNLI是三分类
    '''
    suffix = ["train.tsv", "dev_matched.tsv"]
    if args.test:
        suffix = "test_matched.tsv"
    else:
        suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, on_bad_lines='skip', quoting=3)
    
    examples, chunks = [], []
    for index, row in df.iterrows():
        print(index)
        if args.test:
            label = 0
        else:
            if row["gold_label"] == "entailment":
                label = 2
            elif row["gold_label"] == "neutral":
                label = 1
            else:
                label = 0
        example = MatchExample(str(row['sentence1']), str(row['sentence2']), label)
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(str(row['sentence1'])), PTM_keyword_extractor_yake(str(row['sentence2'])), label)
        chunks.append(chunk)
    return examples, chunks

def read_MNLIMM(input_file, is_training):
    '''0表示contradiction， 1表示neutral，2表示entailment;
        MNLI是三分类
    '''
    suffix = ["train.tsv", "dev_mismatched.tsv"]
    if args.test:
        suffix = "test_mismatched.tsv"
    else:
        suffix = suffix[0] if is_training else suffix[-1] 
    dir_ = os.path.join(input_file, suffix)
    df = pd.read_csv(dir_, sep = '\t', header=0, on_bad_lines='skip', quoting=3)
    
    examples, chunks = [], []
    for index, row in df.iterrows():
        print(index)
        if args.test:
            label = 0
        else:
            if row["gold_label"] == "entailment":
                label = 2
            elif row["gold_label"] == "neutral":
                label = 1
            else:
                label = 0
        example = MatchExample(str(row['sentence1']), str(row['sentence2']), label)
        examples.append(example)
        chunk = ChunkExample(PTM_keyword_extractor_yake(str(row['sentence1'])), PTM_keyword_extractor_yake(str(row['sentence2'])), label)
        chunks.append(chunk)
    return examples, chunks
    
def read_examples(input_file, name, is_training):
    """Read a json file into a list of Example."""
    
    if name == "RTE":
        return read_RTE(input_file, is_training)
    elif name == "MRPC":
        if is_training:
            return read_MRPC_aug(input_file, is_training) # 训练集
        else:
            return read_MRPC(input_file, is_training) # 测试集
    elif name == "MNLIM":
        return read_MNLIM(input_file, is_training)
    elif name == "MNLIMM":
        return read_MNLIMM(input_file, is_training)
    elif name == "QNLI":
        return read_QNLI(input_file, is_training)
    elif name == "QQP":
        return read_QQP(input_file, is_training)
    elif name == "STS-B":
        return read_STSB(input_file, is_training)
    elif name == "SICK":
        if is_training:
            return read_SICK_aug(input_file, is_training)
        else:
            return read_SICK(input_file, is_training)
    
    

def convert_examples_to_features(args, examples, chunks, albert_tokenizer, tokenizer, max_len, is_training):
    features_examples, features_chunks, labels = [], [], []
    
    for (example, chunk) in zip(examples, chunks):
        pair_tokens_example  = tokenizer(example.sentence1, example.sentence2,
                            max_length=args["max_len"],
                            truncation=True,  # 超过最大长度截断
                            padding='max_length',
                             return_tensors="pt",
                            add_special_tokens=True)  # 添加默认的token
        
        features_examples.append(pair_tokens_example)
        
        
        features_chunks.append(0)
        
        labels.append(example.label)
        
    return features_examples, features_chunks, labels
    # return features_examples, features_examples

def get_grad(model):
    for name, parms in model.named_parameters():
        if parms is None or parms.grad is None:continue
        print("-->name:", name, "--grad_requirs:", parms.requires_grad, "--grad_shape:", parms.grad.shape)

if __name__ == "__main__":
    import random
    s =[1,2,1,5,6,7]
    random.shuffle(s)
    print(s)
    