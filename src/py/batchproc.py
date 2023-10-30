from setup import *
from augmenters import *
from helpers import *

# This file contains functions required for application of augmentations to the dataset in a multithreaded environment.


def batch_spec(df):
    """Application of spectorgram calculation to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        result.update({i: build_spec(
            (r['y_s']), sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None)})
    return result


def batch_spec_dist(df):
    """Application of spectorgram calculation and distortion effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        result.update({i: build_spec(bitcrusher_effect(
            r['y_s']), sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None)})
    return result


def batch_spec_delay(df):
    """Application of spectorgram calculation and delay effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        del_time = 0.25  # random.uniform(0.15, 0.4)
        result.update({i: build_spec(simple_delay(r['y_s'], sr=sr, delay_time=del_time, fade=0.6,
                      trim=False), sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None)})
    return result


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
        return room.mic_array.signals[0, :][0:len(y)]


def batch_spec_warp(df):
    """Application of spectorgram calculation and image warping effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        result.update({i: WarpImage_TPS(build_spec(
            r['y_s'], sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None), strength=0.03)})
    return result


def batch_spec_room(df):
    """Application of spectorgram calculation and room simulation effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        len_r = len(r['y_s'])
        result.update({i: build_spec(simulate_room_augm(r['y_s'], sr=sr),
                                     sr, mode="mel", num_filters=186, db=udb, flim=uflim, label=None)})
    return result


def pcc(df, func, n_cores=16):
    """
    This function splits dataframe df into n_cores sub dataframes,
    loads them to n_cores threads and applies func to every one.
    Sub datasets are then combined back into a whole dataframe.
    """
    results = {}
    pool = Pool(n_cores)
    split = np.array_split(df.index.values, n_cores)
    split_dfs = [df[df.index.isin(idx)] for idx in split]
    res = pool.map(func, split_dfs)
    for i in res:
        if i != {}:
            results.update(i)
    pool.close()
    pool.join()
    return results


def pcc_mixup(df, func):
    """
    Similar to pcc, with some differences caused by how mixup technqies work. (In order to prevent target leak,
    Only samples from the same fold should be mixed together. Therefore subset spliting ttechnique is a bit different
    and n_cores is equal to number of folds.
    """
    folds = df.fold.unique()
    n_cores = len(folds)
    results = {}
    pool = Pool(n_cores)
    split = np.array([df[df.fold == f].index.values for f in folds])
    split_dfs = [df[df.index.isin(idx)] for idx in split]
    res = pool.map(func, split_dfs)
    for i in res:
        if i != {}:
            results.update(i)
    pool.close()
    pool.join()
    return results


def batch_spec_mixup(df):
    """Application of spectorgram calculation and audio mixup effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        r2 = df[df.index != i].sample(1)
        y_1 = r['y_s']
        y_2 = r2.y_s.values[0]
        l1 = r.classID
        l2 = r2.classID.values[0]
        y_res, l_res = audio_mixup(y_1, y_2, l1, l2, alpha=0.8)
        result.update({i: [build_spec(y_res, sr, mode="mel",
                      num_filters=186, db=udb, flim=uflim, label=None), l2]})
    return result


def batch_img_spec_mixup(df):
    """Application of spectorgram calculation and image mixup effect to a dataframe sample"""
    result = {}
    for i, r in df.iterrows():
        r2 = df[df.index != i].sample(1)
        im1 = build_spec(r['y_s'], sr, mode="mel",
                         num_filters=186, db=udb, flim=uflim, label=None)
        im2 = build_spec(r2.y_s.values[0], sr, mode="mel",
                         num_filters=186, db=udb, flim=uflim, label=None)
        l2 = r2.classID.values[0]
        result.update({i: [img_mixup(im1, im2, alpha=0.66), l2]})
    return result
