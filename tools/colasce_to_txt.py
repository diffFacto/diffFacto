import numpy as np 
import os 
from anchor_diff.datasets.dataset_utils import load_ply_withNormals
from tqdm import tqdm
def save_data_as_text(data_dir, save_dir):
    fns = sorted(os.listdir(data_dir))  # Get each txt file under the folder according to the path and put it in the list
    # print(fns[0][0:-4])
    ids = [f[:-8] for f in fns if (f[-7:] == 'ori.ply')]
    print(len(ids))
    for i in tqdm(range(len(ids))):
        fn = ids[i]
        point_set = load_ply_withNormals(os.path.join(data_dir, fn + '_ori.ply')).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
        
        seg = np.loadtxt(os.path.join(data_dir, fn + '_ori.labels')).astype(np.long) - 1 + 24# Get the label of the sub category
        mask = np.abs(point_set).sum(1) != 0
        valid_point_set = point_set[mask]
        valid_seg = seg[mask]
        _data = np.concatenate([valid_point_set, valid_seg[:, None]], axis=-1)
        np.savetxt(os.path.join(save_dir, f"{fn}.txt"), _data)
    
save_data_as_text('/orion/u/w4756677/datasets/coalace_data_processed/03636649', '/orion/u/w4756677/datasets/colasce_data_txt/03636649')