import json
import numpy as np


if __name__ == '__main__':
    fnames = ['MlpBlock', 'Linear', 'LinearGeneral', 'SelfAttention', 'EncoderBlock']
    basic_coeff = [4.1, 4., 3.95, 3.4, 3.54]
    b = 2.5

    for fname, coeff in zip(fnames, basic_coeff):
        coeff -= b
        label = json.load(open(f'json/{fname}-label.json', 'r'))
        e = []
        for k in label:
            time = label[k]['time']
            label[k]['e'] = time * (coeff +  0.1 * abs(np.random.randn()))

        json.dump(label, open(f'json/{fname}-label.json', 'w'), indent=2)