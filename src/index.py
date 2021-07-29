import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser

from time import time
from math import ceil
from src.model import *
from multiprocessing import Pool
from src.evaluation.loaders import load_checkpoint

MB_SIZE = 1024

def print_message(*s):
    s = ' '.join(map(str, s))
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

def tok(line):
    docid, cont = line
    d = cleanD(cont, join=False)
    content = ' '.join(d)
    tokenized_content = net.tokenizer.tokenize(content)

    #terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
    #word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
    #terms = [(t, word_indexes.index(idx)) for t, idx in terms]
    #terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]
    terms = list(set([(t, tokenized_content.index(t)) for t in tokenized_content]))
    word_indexes = [-1] + list(range(len(tokenized_content)))
    terms = [(t, word_indexes.index(idx)) for t, idx in terms]
    terms = [(t, idx) for (t, idx) in terms if idx < MAX_LENGTH]

    return tokenized_content, terms, cont, docid # record cont and docid

def quantize(value, scale):
    return int(ceil(value * scale))


def process_batch(super_batch): #directly save results after quantization
    print_message("Start process_batch()", "")
    scale = (1 << 8) / 21.0 # scale = (1<< Quantization_bits) / Max_val

    with torch.no_grad():
        super_batch = list(p.map(tok, super_batch))
        #super_batch = [tok(x) for x in super_batch]

        sorted_super_batch = sorted([(v, idx) for idx, v in enumerate(super_batch)], key=lambda x: len(x[0][0]))
        super_batch = [v for v, _ in sorted_super_batch]
        super_batch_indices = [idx for _, idx in sorted_super_batch]

        #print_message("Done sorting", "")

        every_term_score = []
        contents = []
        docids = []

        for batch_idx in range(ceil(len(super_batch) / MB_SIZE)):
            D = super_batch[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            IDXs = super_batch_indices[batch_idx * MB_SIZE: (batch_idx + 1) * MB_SIZE]
            all_term_scores, cont, docid  = net.index(D, len(D[-1][0])+2)
            every_term_score += zip(IDXs, all_term_scores)
            contents += zip(IDXs, cont)
            docids += zip(IDXs, docid)

        every_term_score = sorted(every_term_score)
        contents = sorted(contents)
        docids = sorted(docids)

        lines = []
        #for _, term_scores in every_term_score:
        #    term_scores = ', '.join([term + ": " + str(int(quantize(score, scale))) for term, score in term_scores])
        #    lines.append(term_scores)
        rets = []
        for idx, term_scores in enumerate(every_term_score):
            _, ts = term_scores
            data = {
                    "id":docids[idx][1],
                    "contents": contents[idx][1],
                    "vector":{}
                    }
            
            for t, s in ts:
                data["vector"][t] = quantize(s, scale)
            rets.append(json.dumps(data) + "\n")
    return rets

if __name__ == "__main__":
    parser = ArgumentParser(description='Eval ColBERT with <query, positive passage, negative passage> triples.')
    
    parser.add_argument('--bsize', dest='bsize', default=128, type=int)
    #parser.add_argument('--triples', dest='triples', default='triples.train.small.tsv')
    #parser.add_argument('--output_dir', dest='output_dir', default='outputs.train/')
    #parser.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])

    parser.add_argument('--collection', default="./baseline_test", type=str)# collection file: tsv, docid \t doc
    parser.add_argument('--output_path', default="./collections/", type=str)
    parser.add_argument('--ckpt', default='./colbert-12layers-max300-32000.dnn',type=str)

    args = parser.parse_args()
    args.input_arguments = args

    print_message("#> Loading model checkpoint.")
    net = MultiBERT.from_pretrained('bert-base-uncased')
    DEVICE = "cuda"
    net = net.to(DEVICE)
    load_checkpoint(args.ckpt, net)
    net.eval()


    p = Pool(16)
    start_time = time()
    g = open(args.output_path+ "/doc0.json", 'w')
    text_id = 0


    with open(args.collection, 'r') as f:

        for idx, passage in enumerate(f):

            if idx % (args.bsize) == 0:
                if idx > 0:
                    plines = process_batch(super_batch)
                    for l in plines:
                        g.write(l)
                throughput = round(idx / (time() - start_time), 1)
                print_message("Processed", str(idx), "passages so far [rate:", str(throughput), "passages per second]")
                super_batch = []

            passage = passage.strip()
            pid, passage = passage.split('\t')
            super_batch.append((pid, passage))
            if idx % 1000000 == 999999 :
                text_id += 1
                print("writen in ", g)
                g = open(args.output_path + "/doc{}.json".format(text_id), "w")

    plines = process_batch(super_batch)
    for l in plines:
        g.write(l)
    g.close()
    f.close()

