import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    
    body_scores_path = "path_body_scores.pkl"
    hand_scores_path = "path_body_scores.pkl"
    leg_scores_path = "path_body_scores.pkl"

    with open(body_scores_path, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(hand_scores_path, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(leg_scores_path, 'rb') as r2:
        r2 = list(pickle.load(r2).items())
  
    right_num = total_num = right_num_5 = 0

    arg.alpha = [1.8, 1.5, 0.5]
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
