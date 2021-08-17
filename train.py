# Example training commands:
#python train.py --config=mlp_mfcc --batch_size=32 --checkpoint_dir=./checkpoints/mlp_baseline_tst
#python train.py --config=resnet_base --batch_size=32 --checkpoint_dir=./checkpoints/resnet_tst
#python train.py --config=resnet_with_augmentation --batch_size=32 --checkpoint_dir=./checkpoints/resnet_aug_tst

#python train.py --config=resnet_with_augmentation --batch_size=32 --checkpoint_dir=./checkpoints/resnet_aug_audioset_tst --train_on_noisy_audioset=True 

import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

sys.path.append('./utils/')
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial    
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle

warnings.filterwarnings('ignore', category=UserWarning)

learning_rate=0.01  # Learning rate.
decay_rate=0.9999  # Learning rate decay per minibatch.
min_learning_rate=0.000001  # Minimum learning rate.

sample_rate = 8000
num_train_steps = 100000

parser = argparse.ArgumentParser()

######## REQUIRED ARGS #########
# Load a preset configuration object. Defines model size, etc. Required
parser.add_argument('--config', type=str, required=True)

# Set a directory to store model checkpoints and tensorboard. Creates a directory if doesn't exist
parser.add_argument('--checkpoint_dir', type=str, required=True)

######## OPTIONAL ARGS #########
# Set batch size. Overrides batch_size set in the config object
parser.add_argument('--batch_size', type=str)

# Default to use GPU. can set to 'cpu' to override
parser.add_argument('--torch_device', type=str, default='cuda')

# Number of processes for parallel processing on cpu. Used mostly for loading in large datafiles
# before training begins or when re-sampling data between epochs
parser.add_argument('--num_workers', type=str, default='8')

# 0.5 unless specified here
parser.add_argument('--dropout_rate', type=str, default='0.5')

# number of batches to accumulate before applying gradients
parser.add_argument('--gradient_accumulation_steps', type=str, default='1')

# include_words flag - if set, data loader will include laughter combined with words
# For example, [laughter - I], [laughter - think], ['laughter -so ']
# This option is not used in the paper
parser.add_argument('--include_words', type=str, default=None)

# Audioset noisy-label training flag
# Flag - if set, train on AudioSet with noisy labels, rather than Switchboard with good labels
parser.add_argument('--train_on_noisy_audioset', type=str, default=None)

args = parser.parse_args()

config = configs.CONFIG_MAP[args.config]
checkpoint_dir = args.checkpoint_dir
batch_size = int(args.batch_size or config['batch_size'])
val_data_text_path = config['val_data_text_path']
feature_fn = partial(config['feature_fn'], sr=sample_rate)
augment_fn = config['augment_fn']
log_frequency = config['log_frequency']
swb_train_audio_pkl_path = config['swb_train_audio_pkl_path']
swb_val_audio_pkl_path = config['swb_val_audio_pkl_path']
audioset_noisy_train_audio_pkl_path = config['audioset_noisy_train_audio_pkl_path']
a_root = config['swb_audio_root']
t_root = config['swb_transcription_root']
expand_channel_dim = config['expand_channel_dim']
torch_device = args.torch_device
num_workers = int(args.num_workers)
dropout_rate = float(args.dropout_rate)
supervised_augment = config['supervised_augment']
supervised_spec_augment = config['supervised_spec_augment']
gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    
if args.include_words is not None:
    include_words = True
else:
    include_words = False
    
if args.train_on_noisy_audioset is not None:
    train_on_noisy_audioset = True
else:
    train_on_noisy_audioset = False

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                        expand_channel_dim=expand_channel_dim)


##################################################################
####################  Setup Training Model  ######################
##################################################################
def load_noise_files():
    noise_files = librosa.util.find_files('./data/background_noise_files/')
    music_files = librosa.util.find_files('./data/background_music_files/')
    np.random.seed(0)
    noise_files += list(np.random.choice(music_files, 200))
    noise_signals = audio_utils.parallel_load_audio_batch(noise_files,n_processes=8,sr=sample_rate)
    noise_signals = [s for s in noise_signals if len(s) > sample_rate]
    return noise_signals

def load_impulse_responses():
    ir_files = librosa.util.find_files('./data/impulse_responses/')
    impulse_responses = audio_utils.parallel_load_audio_batch(ir_files,n_processes=8,sr=sample_rate)
    return impulse_responses

def run_training_loop(n_epochs, model, device, checkpoint_dir,
    optimizer, iterator, log_frequency=25, val_iterator=None, gradient_clip=1.,                
    verbose=True):

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = run_epoch(model, 'train', device, iterator,
            checkpoint_dir=checkpoint_dir, optimizer=optimizer,
            log_frequency=log_frequency, checkpoint_frequency=log_frequency,
            clip=gradient_clip, val_iterator=val_iterator, 
            verbose=verbose)

        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = torch_utils.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            
def run_epoch(model, mode, device, iterator, checkpoint_dir, optimizer=None, clip=None,
                batches=None, log_frequency=None, checkpoint_frequency=None,
                validate_online=True, val_iterator=None, val_batches=None,
                verbose=True):

    """ args:
            mode: 'train' or 'eval'
    """
    
    def _eval_for_logging(model, device, val_itr, val_iterator, val_batches_per_log):
        model.eval()
        val_losses = []; val_accs = []

        for j in range(val_batches_per_log):
            try:
                val_batch = val_itr.next()
            except StopIteration:
                val_itr = iter(val_iterator)
                val_batch = val_itr.next()
                     
            val_loss, val_acc = _eval_batch(model, device, val_batch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
        model.train()
        return val_itr, np.mean(val_losses), np.mean(val_accs)

    def _eval_batch(model,device,batch,batch_index=None,clip=None):
        if batch is None:
            print("None Batch")
            return 0.

        with torch.no_grad():
            seqs, labs = batch
            
            src = torch.from_numpy(np.array(seqs)).float().to(device)
            trg = torch.from_numpy(np.array(labs)).float().to(device)
            output = model(src).squeeze()
            
            criterion = nn.BCELoss()
            bce_loss = criterion(output, trg)
            preds = torch.round(output)
            acc = torch.sum(preds==trg).float()/len(trg) #sum(preds==trg).float()/len(preds)
            
            return bce_loss.item(), acc.item()

    def _train_batch(model,device,batch,batch_index=None,clip=None):

        if batch is None:
            print("None Batch")
            return 0.

        seqs, labs = batch

        src = torch.from_numpy(np.array(seqs)).float().to(device)
        trg = torch.from_numpy(np.array(labs)).float().to(device)

        #optimizer.zero_grad()

        output = model(src).squeeze()
        
        criterion = nn.BCELoss()
        
        preds = torch.round(output)
        acc = torch.sum(preds==trg).float()/len(trg)
        
        bce_loss = criterion(output, trg)
        
        loss = bce_loss
        loss = loss/gradient_accumulation_steps
        loss.backward()

        if model.global_step%gradient_accumulation_steps == 0:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.zero_grad()

        return bce_loss.item(), acc.item()

    if not (bool(iterator) ^ bool(batches)):
        raise Exception("Must pass either `iterator` or batches")

    if mode.lower() not in ['train', 'eval']:
        raise Exception("`mode` must be 'train' or 'eval'")

    if mode.lower() == 'train' and validate_online:
        val_batches_per_epoch =  torch_utils.num_batches_per_epoch(val_iterator)
        val_batches_per_log = int(np.round(val_batches_per_epoch))

        val_itr = iter(val_iterator)

    if mode is 'train':
        if optimizer is None:
            raise Exception("Must pass Optimizer in train mode")
        model.train()
        _run_batch = _train_batch
    elif mode is 'eval':
        model.eval()
        _run_batch = _eval_batch

    epoch_loss = 0

    optimizer = optim.Adam(model.parameters())

    if iterator is not None:
        batches_per_epoch = torch_utils.num_batches_per_epoch(iterator)
        batch_losses = []; batch_accs = []; batch_consistency_losses = []; batch_ent_losses = []
        
        for i, batch in tqdm(enumerate(iterator)):
            # learning rate scheduling
            lr = (learning_rate - min_learning_rate)*decay_rate**(float(model.global_step))+min_learning_rate
            optimizer.lr = lr
                     
            batch_loss, batch_acc = _run_batch(model, device, batch,
                batch_index = i, clip=clip)
                     
            batch_losses.append(batch_loss); batch_accs.append(batch_acc)

            if log_frequency is not None and (model.global_step + 1) % log_frequency == 0:
                val_itr, val_loss_at_step, val_acc_at_step = _eval_for_logging(model, device,
                    val_itr, val_iterator, val_batches_per_log)

                is_best = (val_loss_at_step < model.best_val_loss)
                if is_best:
                    model.best_val_loss = val_loss_at_step

                train_loss_at_step = np.mean(batch_losses)
                train_acc_at_step = np.mean(batch_accs)

                if verbose:
                    print("\nLogging at step: ", model.global_step)
                    print("Train loss: ", train_loss_at_step)
                    print("Train accuracy: ", train_acc_at_step)
                    print("Val loss: ", val_loss_at_step)
                    print("Val accuracy: ", val_acc_at_step)
 
                writer.add_scalar('loss/train', train_loss_at_step, model.global_step)
                writer.add_scalar('acc/train', train_acc_at_step, model.global_step)
                writer.add_scalar('loss/eval', val_loss_at_step, model.global_step)
                writer.add_scalar('acc/eval', val_acc_at_step, model.global_step)
                batch_losses = []; batch_accs = [] # reset

            if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
                state = torch_utils.make_state_dict(model, optimizer, model.epoch,
                                    model.global_step, model.best_val_loss)
                torch_utils.save_checkpoint(state, is_best=is_best, checkpoint=checkpoint_dir)

            epoch_loss += batch_loss
            model.global_step += 1

        model.epoch += 1
        return epoch_loss / len(iterator)

print("Initializing model...")
device = torch.device(torch_device if torch.cuda.is_available() else 'cpu')
print("Using device", device)
model = config['model'](dropout_rate=dropout_rate, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)
torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)
optimizer = optim.Adam(model.parameters())

if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir+'/last.pth.tar', model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    print("Beginning training...")

writer = SummaryWriter(checkpoint_dir)

if augment_fn is not None:
    print("Loading background noise files...")
    noise_signals = load_noise_files()
    augment_fn = partial(augment_fn, noise_signals=noise_signals)
    print("Loading impulse responses...")
    impulse_responses = load_impulse_responses()
    augment_fn = partial(augment_fn, impulse_responses=impulse_responses)
    
if supervised_augment:
    augmented_feature_fn = partial(feature_fn, augment_fn=augment_fn)
else:
    augmented_feature_fn = feature_fn
        
if supervised_spec_augment:
    augmented_feature_fn = partial(feature_fn, spec_augment_fn=audio_utils.spec_augment)
    
#########################################################
############   Do this once, keep in memory  ############
#########################################################

print("Loading switchboard audio files...")
with open(swb_train_audio_pkl_path, "rb") as f: # Loads all switchboard audio files
    switchboard_train_audio_hash = pickle.load(f)

with open(swb_val_audio_pkl_path, "rb") as f:
    switchboard_val_audios_hash = pickle.load(f)

all_audio_files = librosa.util.find_files(a_root,ext='sph')
train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(t_root)
t_files_a, a_files = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
t_files_b, _ = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

all_swb_train_sigs = [switchboard_train_audio_hash[k] for k in switchboard_train_audio_hash if k in a_files]
all_swb_val_sigs = [switchboard_val_audios_hash[k] for k in switchboard_val_audios_hash]

def get_audios_from_text_data(data_file_or_lines, h, sr=sample_rate):
    # This function doesn't use the subsampled offset and duration
    # So it will need to be handled later, in the data loader
    #column_names = ['offset','duration','audio_path','label']
    column_names = ['offset','duration','subsampled_offset','subsampled_duration','audio_path','label']
    audios = []
    if type(data_file_or_lines) == type([]):
        df = pd.DataFrame(data=data_file_or_lines,columns=column_names)
    else:
        df = pd.read_csv(data_file_or_lines,sep='\t',header=None,names=column_names)
        
    audio_paths = list(df.audio_path)
    offsets = list(df.offset)
    durations = list(df.duration)
    for i in tqdm(range(len(audio_paths))):
        aud = h[audio_paths[i]][int(offsets[i]*sr):int((offsets[i]+durations[i])*sr)]
        audios.append(aud)
    return audios

def make_dataframe_from_text_data(data_file_or_lines, h, sr=sample_rate):
    # h is a hash, which maps from audio file paths to preloaded full audio files
    # column_names = ['offset','duration','audio_path','label']
    column_names = ['offset','duration','subsampled_offset','subsampled_duration','audio_path','label']
    if type(data_file_or_lines) == type([]):
        #lines = [l.split('\t') for l in data_file_or_lines]
        df = pd.DataFrame(data=data_file_or_lines,columns=column_names)
    else:
        df = pd.read_csv(data_file_or_lines,sep='\t',header=None,names=column_names)
    return df

def make_text_dataset(t_files_a, t_files_b, audio_files,num_passes=1,
                      n_processes=8,convert_to_text=True,random_seed=None,include_words=False):
    # For switchboard laughter. Given a list of files in a partition (train,val, or test) 
    # extract all the start and end times for laughs, and sample an equal number of negative examples.
    # When making the text dataset, store columns indicating the full start and end times of an event.
    # For example, start at 6.2 seconds and end at 12.9 seconds
    # We store another column with subsampled start and end times (1 per event)
    # and a column with the length of the subsample (typically always 1.0).
    # Then the data loader can have an option to do subsampling every time (e.g. during training) 
    # or to use the pre-sampled times (e.g. during validation)
    # If we want to resample the negative examples (since there are more negatives than positives)
    # then we need to call this function again.
    big_list = []
    assert(len(t_files_a)==len(t_files_b) and len(t_files_a)==len(audio_files))
    for p in range(num_passes):
        lines_per_file = Parallel(n_jobs=n_processes)(
            delayed(dataset_utils.get_laughter_speech_text_lines)(t_files_a[i],
                    t_files_b[i], audio_files[i],convert_to_text,
                    random_seed=random_seed,include_words=include_words) for i in tqdm(range(len(t_files_a))))
        big_list += audio_utils.combine_list_of_lists(lines_per_file)
    return big_list

# By default set up the same number of training examples at a time as are in switchboard
def make_noisy_audioset_text_dataset(audioset_train_files, audioset_train_labels,
                                     audioset_noisy_train_audios_hash, num_lines=35312):
    big_list = []
    for i in range(num_lines):
        ind = np.random.randint(len(audioset_train_labels))
        label = audioset_train_labels[ind]
        f = audioset_train_files[ind]
        file_length = len(audioset_noisy_train_audios_hash[f])/sample_rate
        offset = np.random.uniform(0, file_length-1)
        duration = 1
        line = [offset, duration, offset, duration, f, label]
        big_list.append(line)
    return big_list
        

##################################################################
####################  Setup Validation Data  ######################
##################################################################
val_df = make_dataframe_from_text_data(val_data_text_path, switchboard_val_audios_hash)

val_dataset = data_loaders.SwitchBoardLaughterDataset(
    df=val_df,
    audios_hash=switchboard_val_audios_hash,
    feature_fn=feature_fn,
    batch_size=batch_size,
    sr=sample_rate,
    subsample=False)

val_generator = torch.utils.data.DataLoader(
    val_dataset, num_workers=0, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn)

if train_on_noisy_audioset:
    import audio_set_loading #from audio_set_loading import * #TODO
    with open(audioset_noisy_train_audio_pkl_path, "rb") as f:
        audioset_noisy_train_audios_hash = pickle.load(f)


##################################################################
#######################  Run Training Loop  ######################
##################################################################

while model.global_step < num_train_steps:
    ################## Set up Supervised Training ##################
    #print(f"First time through: {first_time_through}")
    
    print("Preparing training set...")
    lines = make_text_dataset(t_files_a, t_files_b, a_files, num_passes=1, convert_to_text=False, include_words=include_words)
    train_df = make_dataframe_from_text_data(lines, switchboard_train_audio_hash, sr=sample_rate)

    if train_on_noisy_audioset:
        print("Training on noisy Audioset...")
        noisy_audioset_text_lines = make_noisy_audioset_text_dataset(
            audio_set_loading.audioset_train_files, 
            audio_set_loading.audioset_train_labels,
            audioset_noisy_train_audios_hash)
        noisy_audioset_train_df = make_dataframe_from_text_data(noisy_audioset_text_lines,
                                                                audioset_noisy_train_audios_hash, sr=sample_rate)
        train_df = noisy_audioset_train_df
        train_dataset=data_loaders.SwitchBoardLaughterDataset(
            df=train_df,
            audios_hash=audioset_noisy_train_audios_hash,
            feature_fn=augmented_feature_fn,
            batch_size=batch_size,
            sr=sample_rate,
            subsample=True)

    else:
        train_dataset=data_loaders.SwitchBoardLaughterDataset(
            df=train_df,
            audios_hash=switchboard_train_audio_hash,
            feature_fn=augmented_feature_fn,
            batch_size=batch_size,
            sr=sample_rate,
            subsample=True)

    print(f"Number of supervised datapoints: {len(train_dataset)}")
    
    training_generator = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
                                      
    run_training_loop(n_epochs=1, model=model, device=device,
        iterator=training_generator, checkpoint_dir=checkpoint_dir, optimizer=optimizer,
        log_frequency=log_frequency, val_iterator=val_generator,
        verbose=True)
    

