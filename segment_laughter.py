# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
from tqdm import tqdm
import tgt
sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool

sample_rate = 8000

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--threshold', type=str, default='0.5')
parser.add_argument('--min_length', type=str, default='0.2')
parser.add_argument('--input_audio_file', required=True, type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--save_to_audio_files', type=str, default='True')
parser.add_argument('--save_to_textgrid', type=str, default='False')

args = parser.parse_args()


model_path = args.model_path
config = configs.CONFIG_MAP[args.config]
audio_path = args.input_audio_file
threshold = float(args.threshold)
min_length = float(args.min_length)
save_to_audio_files = bool(strtobool(args.save_to_audio_files))
save_to_textgrid = bool(strtobool(args.save_to_textgrid))
output_dir = args.output_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

##### Load the Model

model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
feature_fn = config['feature_fn']
model.set_device(device)

if os.path.exists(model_path):
    torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")
    
##### Load the audio file and features
    
inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
    audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                        expand_channel_dim=config['expand_channel_dim'])

inference_generator = torch.utils.data.DataLoader(
    inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)


##### Make Predictions

probs = []
for model_inputs, _ in tqdm(inference_generator):
    x = torch.from_numpy(model_inputs).float().to(device)
    preds = model(x).cpu().detach().numpy().squeeze()
    if len(preds.shape)==0:
        preds = [float(preds)]
    else:
        preds = list(preds)
    probs += preds
probs = np.array(probs)

file_length = audio_utils.get_audio_length(audio_path)

fps = len(probs)/float(file_length)

probs = laugh_segmenter.lowpass(probs)
instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(args.min_length), fps=fps)

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
            os.path.join(output_dir, fname + '_laughter.TextGrid')))
