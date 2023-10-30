from setup import *

board = Pedalboard([Resample(target_sample_rate=14000),
                    Bitcrush(bit_depth=8),
                    Distortion(drive_db=20),
                    Resample(target_sample_rate=sr)])


def bitcrusher_effect(y, sr=sr):
    """ Distortion effect"""
    y_b = board(y, sr, reset=True)
    return y_b


def simple_delay(y, sr=sr, delay_time=sr, fade=0.7, trim=True):
    """
    With this function we add the Delay effect, which is mixing original signal with its copies delayed in time.
    y - loaded audio data;
    delay_time - as a fraction of the original signal length
    fade - Amplitude of a delayed signal
    trim - if True, we trim the signal to the original length
    """
    delay_time = int(min(len(y)-1, delay_time*len(y)))
    new_l = delay_time + len(y)
    newd = []
    for i in range(new_l):
        if i <= delay_time:
            newd.append(y[i])
        elif i > delay_time and i < len(y):
            newd.append(y[i] + fade*y[i-delay_time])
        else:
            newd.append(fade*y[i-delay_time])
    if trim:
        return np.asarray(newd[0:len(y)])
    else:
        return np.asarray(newd)


def audio_mixup(a, b, l1, l2, alpha=0.5):
    """ Audio Mixup effect
    a - signal 1
    b - signal 2
    l1 - label of signal 1
    l2 - label of singal 2
    alpha - parameter to mix signals at"""
    if len(a) != len(b):
        if len(a) > len(b):
            b = np.pad(b, (0, len(a) - len(b)), 'constant')
        else:
            a = np.pad(a, (0, len(b) - len(a)), 'constant')

    return np.array(alpha*a + (1-alpha)*b), alpha*l1 + (1-alpha)*l2


def img_mixup(img1, img2, alpha=0.8):
    return (img1*alpha+img2*(1-alpha)).astype(np.uint8)


def get_warping_points(im, strength=0.01):
    """
    This function is used to get the warping points for the TPS warping.
    """
    len_x = 224  # np.shape(im)[0]
    len_y = 224  # np.shape(im)[1]
    move = int(strength*((len_x + len_y)/2))
    mx = len(im[0])
    my = len(im)
    zp = np.array([[random.randint(0, int(mx/2)), random.randint(0, int(my/2))], [random.randint(0, int(mx/2)), random.randint(int(my/2), my)],
                   [random.randint(int(mx/2), mx), random.randint(0, int(my/2))], [random.randint(int(mx/2), mx), random.randint(int(my/2), my)]])
    zs = np.array([[min(max(p[0] + random.randint(-move, move), 0), mx),
                  min(max(p[1] + random.randint(-move, move), 0), my)] for p in zp])
    return zp.reshape(-1, len(zp), 2), zs.reshape(-1, len(zs), 2)


def WarpImage_TPS(img, strength=0.1):
    """ Application of TPS warping to an image
    img - image to be warped
    strength - parameter to control the strength of the warping
    In-depth example is provided here: https://github.com/AlanLuSun/TPS-Warp/tree/main
    """
    source, target = get_warping_points(img, strength)
    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = []
    for i in range(0, len(source[0])):
        matches.append(cv2.DMatch(i, i, 0))
    tps.estimateTransformation(target, source, matches)
    new_img = tps.warpImage(img)
    return new_img


def simulate_room(y, sr=sr, room_dim=[10, 20, 10], rt=1.5):
    """Simulate room effect"""
    e_absorption, max_order = pra.inverse_sabine(rt, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=8*sr, use_rand_ism=True, max_rand_disp=0.25, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source([2.5, 2.5, 2.5], signal=y, delay=0.001)
    mic_locs = np.c_[[7.5, 17.5, 7.5]]
    room.add_microphone_array(mic_locs)
    room.simulate(snr=30)
    return room.mic_array.signals[0, :][0:len(y)]
