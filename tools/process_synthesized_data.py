import pickle
import numpy as np 
import os 
from tqdm import tqdm


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

def save_data_as_text(data_dir, save_dir):
    data = pickle.load(open(data_dir, 'rb'))
    pred = data['pred']
    del data['pred']
    idx = [178, 278, 388, 523, 697, 711]
    # remove_top_idx = [3, 24, 43, 114, 123, 131, 135, 139, 195, 201, 209, 219, 235, 282, 335, 366, 384, 385, 390, 403, 420, 
    #                   457, 463, 489, 513, 535, 545, 574, 612, 622, 623, 644, 667, 678, 698, 711, 713, 717, 724, 732]
    
    # preds = [v for k, v in data.items() if 'pred' in k]
    m = 1
    # pred = np.concatenate(preds, axis=0)
    N = pred.shape[0]
    print(N)
    seg = data['pred_seg_mask'] + 12
    # seg = seg[None].repeat(m, axis=0)
    save_data = np.concatenate([pred, seg[..., None]], axis=-1)
    print(save_data.shape)
    # exit()
    save_data = save_data.reshape(N, -1, 4)
    for i in tqdm(range(save_data.shape[0])):
        # if i not in idx: continue
        # print(i)
        # for k, v in data.items():
        #     if 'pred' in k:
        #         print(k)
        #         _data = data[k][i]
        #         _seg = seg[i]
        #         _data = _data[_seg != 8]
        #         _data = np.concatenate([_data, _seg[_seg != 8][..., None]], axis=-1)
        #         print(_data.shape)
        #         np.savetxt(os.path.join(save_dir, f"{i}_sample_{k[-1]}.txt"), _data)
                
        # if np.all(_seg != 0) or np.all(_seg != 1):
        #     print(i)
        #     continue
        np.savetxt(os.path.join(save_dir, f"{i}.txt"), save_data[i])
                
        
    
save_data_as_text('/orion/u/w4756677/diffusion/anchorDIff/weights/ae_lion.pkl', '/orion/u/w4756677/datasets/synthesized_data/03001627')