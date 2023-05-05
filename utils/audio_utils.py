import librosa, numpy as np, pandas as pd, audioread, itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
#from keras.preprocessing.sequence import pad_sequences as keras_pad_seqs
from collections import defaultdict
import text_utils
from sklearn.utils import shuffle
import copy, random
import six
import warnings
import scipy.signal
import pyloudnorm as pyln

warnings.filterwarnings('ignore', category=UserWarning)

"""
# Useful functions for loading audio files
"""

# librosa.load() but return only the signal, not (y, sr)
def librosa_load_without_sr(f, sr=None,offset=None,duration=None):
    if offset is not None and duration is not None:
        return librosa.load(f, sr=sr,offset=offset,duration=duration)[0]
    else:
        return librosa.load(f, sr=sr)[0]

# Runs librosa.load() on a list of files in parallel, returns [y1, y2, ...]
def parallel_load_audio_batch(files,n_processes,sr=None,offsets=None,
    durations=None):
    if offsets is not None and durations is not None:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(files[i],sr=sr,offset=offsets[i],
                duration=durations[i]) for i in tqdm(range(len(files))))
    else:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(f,sr=sr) for f in tqdm(files))

def get_audio_length(path):
    with audioread.audio_open(path) as f:
        return f.duration

"""
# Sequence utils
"""

def keras_pad_seqs(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def pad_sequences(sequences, pad_value=None, max_len=None):
    # If a list of features are supposed to be sequences of the same length
    # But are not, then zero pad the end
    # Expects the sequence length dimension to be the first dim (axis=0)
    # Optionally specify a specific value `max_len` for the sequence length.
    # If none is given, will use the maximum length sequence.
    if max_len is None:
        #lengths = [len(ft) for ft in sequences]
        max_len = max([len(ft) for ft in sequences])
    # Pass along the pad value if provided
    kwargs = {'constant_values': pad_value} if pad_value is not None else {}

    sequences = [librosa.util.fix_length(
        np.array(ft), max_len, axis=0, **kwargs) for ft in sequences]
    return sequences

# This function is for concatenating subfeatures that all have
# the same sequence length
# e.g.  feature_list = [mfcc(40, 12), deltas(40, 12), rms(40, 1)]
# output would be (40, 25)
# The function calls pad_sequences first in case any of the
# sequences of features are off-by-one in length
def concatenate_and_pad_features(feature_list):
    feature_list = pad_sequences(feature_list)
    return np.concatenate(feature_list, axis=1)

"""
# Feature Utils
"""
def featurize_mfcc(f=None, offset=0, duration=None, y=None, sr=None,
        augment_fn=None, hop_length=None, **kwargs):
    """ Accepts either a filepath with optional offset/duration
    or a 1d signal y and sample rate sr. But not both.
    """
    if f is not None and y is not None:
        raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
        raise Exception("Can't use only one of `y` and `sr`")

    if (offset is not None) ^ (duration is not None):
        raise Exception("Can't use only one of `offset` and `duration`")

    if y is None:
        try:
            y, sr = librosa.load(f, sr=sr, offset=offset, duration=duration)
        except:
            import pdb; pdb.set_trace()
    else:
        if offset is not None and duration is not None:
            start_sample = librosa.core.time_to_samples(offset,sr)
            duration_in_samples = librosa.core.time_to_samples(duration,sr)
            y = y[start_sample:start_sample+duration_in_samples]

    # Get concatenated and padded MFCC/delta/RMS features
    S, phase = librosa.magphase(librosa.stft(y, hop_length=hop_length))
    rms = librosa.feature.spectral.rms(S=S).T
    mfcc_feat = librosa.feature.mfcc(y,sr,n_mfcc=13, n_mels=13,
        hop_length=hop_length, n_fft=int(sr/40)).T#[:,1:]
    deltas = librosa.feature.delta(mfcc_feat.T).T
    delta_deltas = librosa.feature.delta(mfcc_feat.T, order=2).T
    feature_list = [rms, mfcc_feat, deltas, delta_deltas]
    feats = concatenate_and_pad_features(feature_list)
    return feats

def featurize_melspec(f=None, offset=None, duration=None, y=None, sr=None,
        hop_length=None , augment_fn=None, spec_augment_fn=None, **kwargs):
    """ Accepts either a filepath with optional offset/duration
    or a 1d signal y and sample rate sr. But not both.
    """
    if f is not None and y is not None:
        raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
        raise Exception("Can't use only one of `y` and `sr`")

    if (offset is not None) ^ (duration is not None):
        raise Exception("Can't use only one of `offset` and `duration`")

    if y is None:
        try:
            y, sr = librosa.load(f, sr=sr, offset=offset, duration=duration)
        except:
            import pdb; pdb.set_trace()
    else:
        if offset is not None and duration is not None:
            start_sample = librosa.core.time_to_samples(offset,sr)
            duration_in_samples = librosa.core.time_to_samples(duration,sr)
            y = y[start_sample:start_sample+duration_in_samples]

    if augment_fn is not None:
        y = augment_fn(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length).T
    S = librosa.amplitude_to_db(S, ref=np.max)
    if spec_augment_fn is not None:
        S = spec_augment_fn(S)
    return S

#def load_audio_file_segments(f, sr, segments):
#    """ Method to load multiple segments of audio from one file. For example,
#    if there are multiple annotations corresponding to different points in the
#    file.
#    Returns: The clipped audio file
#
#    """

def featurize_audio_segments(segments, feature_fn, f=None, y=None, sr=None):
    """ Method to load features for multiple segments of audio from one file.
    For example, if annotations correspond to different points in the file.
    Accepts either a path to an audio file (`f`), or a preloaded signal (`y`)
    and sample rate (`sr`).

    Args:
    segments: List of times in seconds, of the form (offset, duration)
    feature_fn: A function to compute features for each segment
        feature_fn must accept params (f, offset, duration, y, sr)
    f: Filename of audio file for which to get feature
    y: Preloaded 1D audio signal.
    sr: Sample rate
    Returns: A list of audio features computed by feature_fn, for each
        segment in `segments`.
    """

    if f is not None and y is not None:
            raise Exception("You should only pass one of `f` and `y`")

    if (y is not None) ^ bool(sr):
            raise Exception("Can't use only one of `y` and `sr`")

    feature_list = []
    for segment in segments:
        feature_list.append(feature_fn(f=f, offset=segment[0],
            duration=segment[1], y=y, sr=sr))
    return feature_list


"""
# Collate Functions
# For use in Pytorch DataLoaders
# These functions are applied to the list of items returned by the __get_item__
# method in a Pytorch Dataset object.  We need to follow this pattern in order
# to get the benefit of the multi-processing implemented
# in torch.utils.data.DataLoader
"""
def pad_sequences_with_labels(seq_label_tuples, sequence_pad_value=0,
    label_pad_value=None, input_vocab=None, output_vocab=None,
    max_seq_len=None,  max_label_len=None, one_hot_labels=False,
    one_hot_inputs=False, expand_channel_dim=False, auto_encoder_like=False):

    """ Args:
            seq_label_tuples: a list of length batch_size. If the entries in this
            list are already tuples (i.e. type(seq_label_tuples[0]) is tuple),
            we're dealing with the "Basic" setup, where __get_item__ returns 1
            example per file. In that case, we don't need to do anything extra.
            But if seq_label_tuples[0] is a list, then that means we have a
            list of examples for each file, so we need to combine those lists
            and store the results.

            auto_encoder_like: Flag indicating the the labels are also sequences
            like the inputs, and should be padded the same way.

            Pads at the beginning for input sequences and at the end for
            label sequences.

    """
    # First remove any None entries from the list
    # These may have been caused by too short of a sequence in the dataset or some
    # other data problem.
    seq_label_tuples = [s for s in seq_label_tuples if s is not None]

    if len(seq_label_tuples) == 0:
        return None

    try:
      if type(seq_label_tuples[0]) is list:
          # Each file has multiple examples need to combine into one larger
          # list of batch_size*n_samples tuples, instead of a list of lists of tuples
          combined_seq_label_tuples = []
          for i in range(len(seq_label_tuples)):
              combined_seq_label_tuples += seq_label_tuples[i]
          seq_label_tuples = combined_seq_label_tuples
    except:
      import pdb; pdb.set_trace()


    if (output_vocab is None and one_hot_labels) or (input_vocab is None and one_hot_labels):
        raise Exception("Need to provide vocab to convert labels to one_hot.")

    sequences, labels = unpack_list_of_tuples(seq_label_tuples)

    sequences = keras_pad_seqs(sequences, maxlen=max_seq_len, dtype='float32',
        padding='pre', truncating='post', value=sequence_pad_value)

    # Treat the labels the same as the input seqs if this flag is passed
    if auto_encoder_like:
        labels = keras_pad_seqs(labels, maxlen=max_seq_len, dtype='float32',
        padding='pre', truncating='post', value=sequence_pad_value)

    # If there are no labels, then expect the batch of labels as [None, None...]
    elif labels[0] is not None:
        if label_pad_value is not None:
            # label_pad_value should be the string value, not the integer in the voc
            labels = pad_sequences(labels, label_pad_value, max_len=max_label_len)

        # Convert vocab to integers after padding
        if output_vocab is not None:
            labels = [text_utils.sequence_to_indices(l, output_vocab) for l in labels]

        if one_hot_labels:
            labels = [text_utils.np_onehot(l, depth=len(output_vocab)) for l in labels]

    if expand_channel_dim:
        sequences = np.expand_dims(sequences, 1)
        if auto_encoder_like:
            labels = np.expand_dims(labels, 1)
    return sequences, labels

"""
# Data Augmentation Functions
"""
# Speed up or slow down audio by `factor`. If factor is >1,
# we've sped up, so we need to pad. If factor <1, take the beginning of y
# set length of new_y to match length of y
def set_length(new_y, y):
    if len(new_y) < len(y):
        new_y = librosa.util.fix_length(new_y, len(y))
    elif len(new_y) > len(y):
        new_y = new_y[0:len(y)]
    return new_y

def random_speed(y, sr, prob=0.2, min_speed=0.9, max_speed=1.1):
    if np.random.uniform(0,1) < prob:
        factor = np.random.uniform(min_speed, max_speed)
        new_sr = sr*factor
        new_y = librosa.core.resample(y,sr,new_sr)
        return set_length(new_y, y)
    else:
        return y

def random_stretch(y, sr, prob=0.2, min_stretch=0.9, max_stretch=1.1):
    if np.random.uniform(0,1) < prob:
        factor = np.random.uniform(min_stretch, max_stretch)
        new_y = librosa.effects.time_stretch(y, factor)
        return set_length(new_y, y)
    else:
        return y

def random_pitch(y, sr, prob=0.2, min_shift=-4, max_shift=5, bins_per_octave=24):
    if np.random.uniform(0,1) < prob:
        steps = np.random.randint(min_shift,max_shift)
        new_y = librosa.effects.pitch_shift(y, sr, n_steps=steps, bins_per_octave=bins_per_octave)
        return set_length(new_y, y)
    else:
        return y

def random_noise(y, sr, noise_signals, min_snr=6, max_snr=30, prob=1.):
    if np.random.uniform(0,1) < prob:
        meter = pyln.Meter(sr)
        snr = np.random.uniform(min_snr, max_snr)
        noise_signal = np.random.choice(noise_signals)
        if len(noise_signal) < len(y):
            raise Exception("length of the background noise signal is too short")
        noise_start = int(np.random.uniform(0, len(noise_signal)-len(y)))
        noise = noise_signal[noise_start:noise_start+len(y)]
        sig_loudness = meter.integrated_loudness(y)
        noise_loudness = meter.integrated_loudness(noise)
        loudness_normalized_noise = pyln.normalize.loudness(noise, noise_loudness, sig_loudness-snr)
        # Compute and adjust snr
        combined_sig = y + loudness_normalized_noise
        return combined_sig
    else:
        return y
    
# Sample rate must be the same between signal and IR
# Based on https://github.com/mravanelli/pySpeechRev
def conv_reverb(signal_clean, IR):
    def _shift(xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e
    
    # Signal normalization
    signal_clean=signal_clean/np.max(np.abs(signal_clean))

    # IR normalization
    IR=IR/np.abs(np.max(IR))
    p_max=np.argmax(np.abs(IR))

    # Convolve
    signal_rev=scipy.signal.fftconvolve(signal_clean, IR, mode='full')

    # Normalization
    signal_rev=signal_rev/np.max(np.abs(signal_rev))
    # IR delay compensation
    signal_rev=_shift(signal_rev, -p_max)
    # Cut reverberated signal (same length as clean sig)
    signal_rev=signal_rev[0:signal_clean.shape[0]]
    return signal_rev

def random_reverb(y, sr, impulse_responses):
    IR = np.random.choice(impulse_responses)
    return conv_reverb(y, IR)

def random_augment(y, sr, noise_signals, impulse_responses):
    #functions = shuffle([random_speed, random_stretch, random_pitch,
    functions = shuffle([random_speed, random_stretch, random_pitch,
        partial(random_noise, noise_signals=noise_signals),
        partial(random_reverb, impulse_responses=impulse_responses)])
    for fn in functions:
        y = fn(y, sr=sr)
    #fn = np.random.choice(functions)
    #y = fn(y, sr=sr, prob=1.)
    return y

def random_augment_strong(y, sr, noise_signals, impulse_responses):
    #functions = shuffle([random_speed, random_stretch, random_pitch,
    functions = shuffle([
        partial(random_stretch,prob=1.,min_stretch=0.5,max_stretch=1.7),
        partial(random_speed,prob=1.,min_speed=0.5,max_speed=1.7),
        partial(random_pitch,prob=1.,min_shift=-18,max_shift=18),
        partial(random_noise, noise_signals=noise_signals, min_snr=1, max_snr=8),
        partial(random_reverb, impulse_responses=impulse_responses),
        partial(random_noise, noise_signals=noise_signals, min_snr=1, max_snr=8),
        partial(random_reverb, impulse_responses=impulse_responses)])
    for fn in functions:
        y = fn(y, sr=sr)
    return y

def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = copy.deepcopy(spec)
    num_mel_channels = cloned.shape[0]
    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0,np.maximum(1,num_mel_channels - f))
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f):
            return cloned
        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()
    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = copy.deepcopy(spec)
    len_spectro = cloned.shape[1]
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, np.maximum(1, len_spectro - t))
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t):
            return cloned
        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[:, t_zero:mask_end] = cloned.mean()
    return cloned


def spec_augment(spec, prob=1.):
    # Adapted from https://github.com/zcaceres/spec_augment
    if np.random.uniform(0, 1) < prob:
        return freq_mask(time_mask(spec))
    else:
        return time_mask(freq_mask(spec))

"""
def time_warp(spec, W=5):
    num_rows = spec.shape[0]
    spec_len = spec.shape[1]
    y = num_rows//2
    horizontal_line_at_ctr = spec[y]
    assert len(horizontal_line_at_ctr) == spec_len
    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)
    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)
"""




"""
# Misc Functions
"""
def unpack_list_of_tuples(list_of_tuples):
    return [list(tup) for tup in list(zip(*list_of_tuples))]

def combine_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def reverse_sequence(feature_seq):
    return np.flip(feature_seq, axis=0)

def reverse_sequence_batch(batch_feats):
    for i in range(len(batch_feats)):
        batch_feats[i] = reverse_sequence(batch_feats[i])
    return batch_feats

def dedup_list(l):
    l = copy.deepcopy(l)
    i = 1
    while i < len(l):
        if l[i] == l[i-1]:
            del l[i]
        else:
            i += 1
    return l

def times_overlap(start1, end1, start2, end2):
    if end1 < start2 or end2 < start1:
        return False
    else:
        return True

def start_end_to_offset_duration(start, end):
    return start, end-start

def subsample_time(offset, duration, audio_file_length, subsample_length=1., padding_length=0.5, random_seed=None):
    start_time = np.maximum(0, offset-padding_length)
    end_time = np.minimum(offset+duration+padding_length, audio_file_length)
    if random_seed is not None:
        np.random.seed(random_seed)
    start = np.maximum(start_time,np.random.uniform(start_time, end_time-subsample_length))
    return (start, subsample_length)
