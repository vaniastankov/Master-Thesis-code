# This .py file describes the application of augmentation techniques to the dataset.a3
# In order to save time and utilize resources more efficiently, we use multiprocessing for this part.
# Code from this file is used in the preprocesing notebook.

from setup import *
from augmenters import *
from helpers import *

#Room simulation parameters
room_dim = [10, 10, 5]
rt = 0.25
e_absorption, max_order = pra.inverse_sabine(rt, room_dim)

def simulate_room_augm(y, sr=sr, trim=False, max_order=max_order, tol=0.2):
    """ This function is similar to the one found in augmenters, with core difference being more efficient variable handling for multithraded application."""
    room = pra.ShoeBox(
        room_dim, fs=8*sr, use_rand_ism=True, max_rand_disp=0.25, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source([2.5, 2.5, 3.5], signal=y, delay=0)
    mic_locs = np.c_[[7.5, 7.5, 4.5]]
    room.add_microphone_array(mic_locs)
    room.simulate(snr=30)
    if trim:
        return room.mic_array.signals[0, :][0:len(y)]
    else:
        n_l = min(len(room.mic_array.signals[0, :]), int((1+tol)*len(y)))
        return room.mic_array.signals[0, :][0:n_l]


def spc_wrap(y,sr):
    """This function is used to wrap the spectrogram calculation function for multithreaded application."""
    return build_spec(y, sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None)

def batch_func_application(df, func_name):
    """This function is used to apply augmentation function to a dataframe"""
    result = {}
    if func_name == "spectrogram":
        for i, r in df.iterrows():
            result.update({i: spc_wrap(r['y_s'],sr)})
    elif func_name == "distortion":
        for i, r in df.iterrows():
            result.update({i: spc_wrap(bitcrusher_effect(r['y_s']), sr)})
    elif func_name == "delay":
        for i, r in df.iterrows():
            result.update({i: spc_wrap(simple_delay(r['y_s'], sr=sr, delay_time=0.25, fade=0.6,
            trim=False), sr)})
    elif func_name == "warping":
        for i, r in df.iterrows():
            result.update({i: warp_img(spc_wrap(r['y_s'], sr), strength=0.03)})
    elif func_name == "room":
        for i, r in df.iterrows():
            result.update({i: spc_wrap(simulate_room_augm(r['y_s'], sr=sr),sr)})
    elif func_name == "audio_mixup":
        for i, r in df.iterrows():
            r2 = df[df.index != i].sample(1)
            y_1 = r['y_s']
            y_2 = r2.y_s.values[0]
            l1 = r.classID
            l2 = r2.classID.values[0]
            y_res, l_res = audio_mixup(y_1, y_2, l1, l2, alpha=0.8)
            result.update({i: [spc_wrap(y_res, sr), l2]})
    elif func_name == "img_mixup":
        for i, r in df.iterrows():
            r2 = df[df.index != i].sample(1)
            y_1 = r['y_s']
            y_2 = r2.y_s.values[0]
            l1 = r.classID
            l2 = r2.classID.values[0]
            y_res, l_res = audio_mixup(y_1, y_2, l1, l2, alpha=0.8)
            result.update({i: [spc_wrap(y_res, sr), l2]})
    else:
        print("Wrong function name!")
        return None
    return result

def pcc(df, func_name, n_cores=16):
    """
    This function splits dataframe df into n_cores sub dataframes,
    loads them to n_cores threads and applies augmentation function with func_name to every piece.
    Sub datasets are then combined back into a whole dataframe.
    """
    if func_name in ["spectrogram", "distortion", "delay", "warping", "room"]:
        results = {}
        pool = Pool(n_cores)
        split = np.array_split(df.index.values, n_cores)
        split_dfs = [df[df.index.isin(idx)] for idx in split]
        partial_batch_func = partial(batch_func_application, func_name=func_name)
        res = pool.map(partial_batch_func, split_dfs)
        for i in res:
            if i != {}:
                results.update(i)
        pool.close()
        pool.join()
        return results
    else:
        print("Plese use pcc_mixup function for this augmentation technique.")
        return None


def pcc_mixup(df, func_name):
    """
    Similar to pcc, with some differences caused by how mixup technqies work. (In order to prevent target leak,
    Only samples from the same fold should be mixed together. Therefore subset spliting ttechnique is a bit different
    and n_cores is equal to number of folds.
    """
    if func_name in ["audio_mixup","img_mixup"]:
        folds = df.fold.unique()
        n_cores = len(folds)
        results = {}
        pool = Pool(n_cores)
        split = np.array([df[df.fold == f].index.values for f in folds])
        split_dfs = [df[df.index.isin(idx)] for idx in split]
        partial_batch_func = partial(batch_func_application, func_name=func_name)
        res = pool.map(partial_batch_func, split_dfs)
        for i in res:
            if i != {}:
                results.update(i)
        pool.close()
        pool.join()
        return results
    else:
        print("Plese use pcc function for this augmentation technique.")
        return None