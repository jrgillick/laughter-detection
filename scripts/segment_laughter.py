import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
from tqdm import tqdm
import tgt
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
sys.path.append('../')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool

sample_rate = 8000

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--threshold', type=str, default='0.5')
parser.add_argument('--min_length', type=str, default='0.2')
parser.add_argument('--input_audio_file', type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--save_to_audio_files', type=str, default='True')
parser.add_argument('--save_to_textgrid', type=str, default='False')

args = parser.parse_args()

# python segment_laughter.py --model_path=/mnt/data0/jrgillick/projects/laughter-detection/checkpoints/resnet_43fps_spec_augment_d01 --config=resnet_43fps_spec_augment --input_audio_file=/mnt/data0/jrgillick/projects/laughter-detection/data/kimiko/2017.07.18.1130/01.wave.m4a

#'/mnt/data0/jrgillick/projects/laughter-detection/checkpoints/resnet_43fps_spec_augment_d01'
#config = configs.CONFIG_MAP['resnet_43fps_spec_augment']

model_path = args.model_path
config = configs.CONFIG_MAP[args.config]
audio_path = args.input_audio_file
threshold = float(args.threshold)
min_length = float(args.min_length)
min_length = laugh_segmenter.seconds_to_frames(min_length, fps=43)
save_to_audio_files = bool(strtobool(args.save_to_audio_files))
save_to_textgrid = bool(strtobool(args.save_to_textgrid))
output_dir = args.output_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Load the Model

model = config['model'](dropout_rate=0., linear_layer_size=config['linear_layer_size'])
feature_fn = config['feature_fn']
model.set_device(device)

if os.path.exists(model_path):
    torch_utils.load_checkpoint(model_path, model)
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")
    
##### Load the audio file and features
    
inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
    audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                        expand_channel_dim=config['expand_channel_dim'])

inference_generator = torch.utils.data.DataLoader(
    inference_dataset, num_workers=4, batch_size=64, shuffle=False, collate_fn=collate_fn)


##### Make Predictions

probs = []
for model_inputs, _ in tqdm(inference_generator):
    x = torch.from_numpy(model_inputs).float().to(device)
    preds = list(model(x).cpu().detach().numpy().squeeze())
    probs += preds
probs = np.array(probs)

filtered = laugh_segmenter.lowpass(probs)
instances = laugh_segmenter.get_laughter_instances(filtered, threshold=threshold, min_length=min_length, fps=43)

print(); print("found %d laughs." % (len (instances)))

if len(instances) > 0:
    full_res_y, full_res_sr = librosa.load(audio_path,sr=44100)
    wav_paths = []
    maxv = np.iinfo(np.int16).max
    
    if save_to_audio_files:
        if output_dir is None:
            raise Exception("Need to specify an output directory to save audio files")
        else:
            os.system(f"mkdir -p {output_dir}")
            for index, instance in enumerate(instances):
                laughs = laugh_segmenter.cut_laughter_segments([instance],full_res_y,full_res_sr)
                wav_path = output_dir + "/laugh_" + str(index) + ".wav"
                scipy.io.wavfile.write(wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                wav_paths.append(wav_path)
            print(laugh_segmenter.format_outputs(instances, wav_paths))
    
    if save_to_textgrid:
        laughs = [{'start': i[0], 'end': i[1]} for i in instances]
        tg = tgt.TextGrid()
        laughs_tier = tgt.IntervalTier(name='laughter', objects=[
        tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
        tg.add_tier(laughs_tier)
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        tgt.write_to_file(tg, os.path.join(output_dir, fname + '_laughter.TextGrid'))

        print('Saved laughter segments in {}'.format(
            os.path.join(output_path, fname + '_laughter.TextGrid')))