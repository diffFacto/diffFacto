import pickle 
from six.moves import cPickle
import os.path as osp

def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: a generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """

    in_file = open(file_name, "rb")
    if python2_to_3:
        size = cPickle.load(in_file, encoding="latin1")
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding="latin1")
        else:
            yield cPickle.load(in_file)
    in_file.close()


def process():
    (
        game_data,
        word2int,
        int2word,
        int2sn,
        sn2int,
        sorted_sn,
    ) = unpickle_data("/mnt/disk3/wang/diffusion/datasets/partglot_data/game_data.pkl")
    print(len(word2int))
    with open("/mnt/disk3/wang/diffusion/anchorDIff/python/anchor_diff/models/encoders/language_utils/word2int.pkl", "wb") as f:
        pickle.dump(word2int, f)
    with open("/mnt/disk3/wang/diffusion/anchorDIff/python/anchor_diff/models/encoders/language_utils/int2word.pkl", "wb") as ff:
        pickle.dump(int2word, ff)

process()