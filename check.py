import numpy as np

def check(filename, vec, DEBUG = False):
    with open(filename, 'r') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    data_vec = []
    for d in data:
        data_vec.append(list(map(float, d.split(' ')[2:])))


    ans_ = []

    if DEBUG:
        print('-----------------------')
    for _data_vec in data_vec:
        ans = 0
        
        for a, b in zip(vec, _data_vec):
            ans += a * b
        ans_.append(ans)
        if DEBUG:
            print(np.argsort(ans_))
    if DEBUG:
        print('-----------------------')
    return np.argsort(ans_)[-1] // 3 if np.argsort(ans_)[-1] < 0.4 else 3
