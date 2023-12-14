#This .py file contains all supporting functions that are used in ..ipynb notebooks.a3
#Expect functions like plotting, audio processing, etc to be located here.
from setup import *



def plot_waveform(y, sr, label=None):
    """
    This function is used to build a Waveform view of the signal.
    Audio recording has to be provided as an array y, its samplerate is also required in order to calculate the time axis correctly.
    y - the audio signal (array)
    sr - the sampling rate
    """
    tick_freq = 3
    dpi = 256
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    plt.plot(y)
    plt.xticks(np.arange(0, len(y), tick_freq*sr),
               tick_freq*np.arange(0, len(y)/(tick_freq*sr)))
    buf = io.BytesIO()
    if label == None:
        fig.axis('off')
        img.colorbar.remove()
        fig.savefig(buf, format="png", dpi=dpi)
    else:
        fig.set_tight_layout({"pad": .25})
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        fig.savefig(plot_directory+f"Waveform {label}.png", format="png", dpi=dpi)
        plt.title(label + " Waveform")
        fig.savefig(buf, format="png", dpi=dpi)
    plt.close()
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def build_spec(y, sr, mode="mel", num_filters=128, db=('auto', 'auto'), flim=('auto', 'auto'), label=None):
    """
    This function builds a spectrogram from a given audio signal. Here are the parameters:
        y - the audio signal (1/2 dimensional array)
        sr - the sampling rate (number of samples per second)
        mode - the type of spectrogram. Can be either "mel" or "log"
        num_filters - the number of filters to use for the spectrogram, has different effect depending on the mode
        db - the decibel range to use for the spectrogram
        flim - the frequency range to use for the spectrogram
        label - the label of the spectrogram. If None, the spectrogram is returned as an image array, if exists, the spectrogram is saved in the plot directory
    """
    dpi = 56 if label == None else 512  # Image size varies depending on use, models use 224x224 images, but to view in a notebook, bigger images are needed
    # dpi = 56 if label == None else 56 * 3  # Image size varies depending on use, models use 224x224 images, but to view in a notebook, bigger images are needed
    fig, ax = plt.subplots(figsize=(4 if label == None else 5, 4), dpi=dpi)
    fmin = 0 if flim[0] == 'auto' else flim[0]
    fmax = sr/2 if flim[1] == 'auto' else flim[1]
    dbmin = -100 if db[0] == 'auto' else db[0]
    dbmax = 0 if db[1] == 'auto' else db[1]
    if mode == "mel":
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_filters)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB, sr=sr, ax=ax, x_axis='time', y_axis='mel')
    elif mode == "log":
        S = np.abs(librosa.stft(
            y, hop_length=num_filters//4, n_fft=num_filters))**2
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_dB, hop_length=num_filters//4, sr=sr, ax=ax, x_axis='time', y_axis='log')
    plt.ylim([fmin, fmax])
    img.set_clim(dbmin, dbmax)
    fig.colorbar(img, ax=ax, format='%2.0fdB')
    prefix = " Mel" if mode == "mel" else " Log STFT"
    buf = io.BytesIO()
    if label == None:
        ax.axis('off')
        img.colorbar.remove()
        fig.tight_layout(pad=0)
        fig.savefig(buf, format="png", dpi=dpi,
                    pad_inches=0, bbox_inches='tight')
    else:
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        fig.set_tight_layout({"pad": .25})
        fig.savefig(plot_directory+f"{label}.png",
                    format="png", dpi=dpi, bbox_inches='tight')
        ax.set_title(label + prefix + " Spectrogram")
        fig.savefig(buf, format="png", dpi=dpi,
                    pad_inches=0, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    # img = image_from_plot#np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_room(room, markersize=100, img_order=0):
    """
    This is a slightly enhanced version of the plot_room function from pyroomacoustics library
    Here we can specify the aspect ratio of the plot and source and mic marker sizes.
    room - the pyroomacoustics room object
    markersize - the size of the markers for sources and microphones
    img_order - somewhat of unclear parameter, but it affect placement of objects in the room
    """
    room_dim = [int(max(w)) for w in room.walls[0].corners]
    fig = plt.figure(figsize=(5, 4))
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    ax.set_aspect('auto', adjustable='box')
    ax.set_box_aspect(aspect=None)
    ax.set_xlim3d(0, 0.5*max(room_dim))
    ax.set_ylim3d(0, max(room_dim))
    ax.set_zlim3d(0, 0.5*max(room_dim))
    fig.add_axes(ax)
    # plot the walls
    for w in room.walls:
        tri = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.5)
        tri.set_color(colors.rgb2hex(np.random.rand(3)))
        tri.set_edgecolor("k")
        ax.add_collection3d(tri)
    # define some markers for different sources and colormap for damping
    markers = ["o", "s", "v", "."]
    cmap = plt.get_cmap("viridis")
    # draw the scatter of images
    for i, source in enumerate(room.sources):
        # draw source
        ax.scatter(
            source.position[0],
            source.position[1],
            source.position[2],
            c=[cmap(1.0)],
            s=20,
            marker=markers[i % len(markers)],
            edgecolor=cmap(1.0),
        )
        # draw images
        if img_order is None:
            img_order = room.max_order

        I = source.orders <= img_order
        if len(I) > 0:
            has_drawn_img = True
        # plot the images
        ax.scatter(
            source.images[0, I],
            source.images[1, I],
            source.images[2, I],
            c=cmap(0),
            s=markersize,
            marker=markers[i % len(markers)],
            edgecolor=cmap(0),
        )
    mic_marker_size = markersize
    # draw the microphones
    if room.mic_array is not None:
        for i in range(room.mic_array.nmic):
            ax.scatter(
                room.mic_array.R[0][i],
                room.mic_array.R[1][i],
                room.mic_array.R[2][i],
                marker="x",
                linewidth=0.5,
                s=mic_marker_size,
                c="k",
            )
    # ax.scatter(room_dim[1], room_dim[0], room_dim[2], c='red', marker='*', s=1000)
    plt.tight_layout()
    plt.show()


def plot_specs(df, label=None, clss=None, from_y=False):
    """
    This function is used to plot multiple spectrograms at once. Used in preprocessing notebook to illustrate every class in the dataset.
    """
    fig, axs = plt.subplots(2, 5, figsize=(16, 8), dpi=56*3)
    for i, cls in enumerate(clss.keys()):
        axs[i // 5][i % 5].axis('off')
        if from_y:
            img = build_spec(df[df.slice_file_name == clss[cls]]['y_s'].values[0], sr, mode="mel",
                             num_filters=186, db=(-100, 10), flim=(20, 20000), label=cls)
        else:
            img = df[df.slice_file_name == clss[cls]]['r'].values[0]
            axs[i // 5][i % 5].set_title(cls, fontsize=16)
        axs[i // 5][i % 5].imshow(img)
    if label:
        plt.savefig(plot_directory + label + ".png")
        fig.suptitle(label, fontsize=24)
    fig.subplots_adjust(hspace=0.04, wspace=0.2)
    plt.show()


def plot_sample(y, sr, label="", include_audio=True):
    """
    This function is used to plot a waveform and spectrogram of a given audio signal.
    Used in motivation notebook to describe audio data in deep detail.
    y - the audio signal (array)
    sr - the sampling rate
    label - the label of the spectrogram
    include_audio - whether to include the audio file as well (for in-notebook previews)
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 8), dpi=224)
    axs[0].axis('off')
    axs[0].imshow(plot_waveform(y, sr, label=label), interpolation='nearest')
    axs[1].axis('off')
    axs[1].imshow(build_spec(y, sr, mode="mel", num_filters=186, db=(-100,
                  10), flim=(20, 20000), label=label), interpolation='nearest')
    plt.show()
    if include_audio:
        IPython.display.display(IPython.display.Audio(data=y, rate=sr))


def get_prf1_table(df, names=[], metric='precision', avg='macro'):
    """
    This function calculates the precision/recall/f1 metric table based on a perticular dataframe.
    It is expected that df has fold, class, and prediction columns.
    Output is a dataframe with metric value for each fold and each augmentation technique.
    Metrics are averaged over class.
    """
    t = df[df.source == 'o'].copy()  # Only consider original samples
    res = pd.DataFrame(columns=names)
    for fold in df.fold.unique():
        sdf = t[t['fold'] == fold]
        r = []
        for name in names:
            if metric == 'precision':
                r.append(precision_score(sdf['class'], sdf[name], average=avg))
            elif metric == 'recall':
                r.append(recall_score(sdf['class'], sdf[name], average=avg))
            elif metric == 'f1':
                r.append(f1_score(sdf['class'], sdf[name], average=avg))
            else:
                r.append(-1)
        res.loc[fold] = r
    return res


def get_accuracy_table(df, names=[]):
    """
    This function calculates the accuracy metric based on a perticular dataframe.
    It is expected that df has fold, class, and prediction columns.
    Output is a dataframe with metric value for each fold and each augmentation technique.
    Metrics are averaged over class.
    """
    t = df[df.source == 'o'].copy()  # Only consider original samples
    for name in names:
        t[name] = (t[name] == t['class']).apply(int)

    res = pd.DataFrame(columns=names)
    for fold in df.fold.unique():
        group = t[t['fold'] == fold].groupby(
            'class').agg({name: 'mean' for name in names})
        res.loc[fold] = group.mean()
    return res


def build_confusion_matrix(df, name, prefix = ""):
    df = df[df.source == 'o'].copy()
    """
    This function builds a confusion matrix for a given dataframe and augmentation technique-based predictions (stored as a column in it).
    OneHotEncoder is used to convert class names to numbers. and correctly represent them on a plot.
    Note that it is possible to build this plot without it, yet it is handy to ensure correct label order on a plot.
    """
    enc = OneHotEncoder(handle_unknown='ignore',
                        feature_name_combiner=lambda x, y: y)
    enc.fit(df['class'].to_numpy().reshape(-1, 1))
    r = []
    for fold in df.fold.unique():
        cm = confusion_matrix(
            enc.transform(df['class'].to_numpy(
            ).reshape(-1, 1)).toarray().argmax(axis=1),
            enc.transform(df[name].to_numpy().reshape(-1, 1)
                        ).toarray().argmax(axis=1),
            normalize='true'
        )
        r.append(cm)
    cm = np.round(np.mean(r, axis=0),2)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=enc.get_feature_names_out())
    dpi = 56*4
    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    disp.plot(cmap=plt.cm.bone_r, ax=ax, colorbar=False,
              xticks_rotation='vertical')
    buf = io.BytesIO()
    fig.set_tight_layout({"pad": .25})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.savefig(
        plot_directory+f"Confusion Matrix for {name} ({prefix}).png", format="png", dpi=dpi, bbox_inches='tight')
    ax.set_title(f'Confusion Matrix for experiment {name} ({prefix})', fontsize=18)
    fig.savefig(buf, format="png", dpi=dpi,
                pad_inches=0, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


bp_names_lookup = {
    'dist': 'Distortion',
    'mixup': 'Audio\nMixup',
    'imixup': 'Image\nMixup',
    'room': 'Room\nSim.',
    'warp': 'Image\nWarping',
    'delay': 'Delay',
    'spectrum': 'Original\nOnly',
    'image_only': 'Image\nComb.',
    'audio_only': 'Audio\nComb.',
    'audio_only_nd': 'Audio\nw/o Delay',
    'all': 'All Comb.'
}

def build_boxplot(df, names=[], label=''):
    """In this function we build a boxplot based on a metric table,
    input dataframe is expected to store a metric value for each fold and each dataset version.
    Metrics can be any numerical metric (e.g. accuracy, precision, recall, f1-score, etc.)
    """
    bpd = [df[name] for name in names]
    dpi = 512
    fig, ax = plt.subplots(figsize=(16, 8), dpi=dpi)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    bp = ax.boxplot(bpd, labels=[bp_names_lookup[n] for n in names],whis = [0,100], patch_artist=True,
                    meanline=True, showmeans=True, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#A1B7BB')
    for whisker in bp['whiskers']:
        whisker.set(color='#481A76',
                    linewidth=1.5,
                    linestyle="--")
    for cap in bp['caps']:
        cap.set(color='#090920',
                linewidth=3)
    for median in bp['medians']:
        median.set(color='#F09E73',
                   linewidth=3)
    for median in bp['means']:
        median.set(color='#D55B67',
                   linewidth=3)

    fig.legend([bp['means'][0], bp['medians'][0]], ['Mean', 'Median'],bbox_to_anchor=(0.99,0.95), prop={'size': 18})

    buf = io.BytesIO()
    fig.set_tight_layout({"pad": .25})
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.savefig(
        plot_directory+f"Box Plot for {label}.png", format="png", dpi=dpi, wrap = True)
    ax.set_title(label, fontsize=24)
    fig.savefig(buf, format="png", dpi=dpi,
                pad_inches=0, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def summarize_table_to_latex(df, name, names = []):
    """This function is used to summarize a metric table into a latex table.
    A handy way to include it in a thesis. Is especially beneficial for bigger tables.
    """
    t = df[names].copy()
    stats = pd.concat([t.mean(), t.median(), t.std()], axis=1).T
    t = pd.concat([t, stats], axis=0)
    t.rename(columns=bp_names_lookup, inplace=True)
    lt = t.to_latex(index=True, float_format="%.2f")
    with open("../Data/"+name+"_table.txt" ,'w') as f:
        f.write(lt)

