#python train.py --config=mlp_mfcc --batch_size=32 --checkpoint_dir=/mnt/data0/jrgillick/projects/laughter-detection/checkpoints/mlp_baseline_b32
import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
sys.path.append('../')
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial    
from tensorboardX import SummaryWriter

learning_rate=0.001  # Learning rate.
decay_rate=0.9999  # Learning rate decay per minibatch.
min_learning_rate=0.000001  # Minimum learning rate.


parser = argparse.ArgumentParser()

# Path to store the parsed label times and inputs for Switchboard
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--batch_size', type=str)
parser.add_argument('--checkpoint_dir', type=str, required=True)
parser.add_argument('--torch_device', type=str, default='cuda')
parser.add_argument('--num_workers', type=str, default='8')
parser.add_argument('--dropout_rate', type=str, default='0.5')

args = parser.parse_args()

config = configs.CONFIG_MAP[args.config]

batch_size = int(args.batch_size or config['batch_size'])
feature_fn = config['feature_fn']
augment_fn = config['augment_fn']
train_data_text_path = config['train_data_text_path']
val_data_text_path = config['val_data_text_path']
log_frequency = config['log_frequency']
swb_train_audio_pkl_path = config['swb_train_audio_pkl_path']
swb_val_audio_pkl_path = config['swb_val_audio_pkl_path']
checkpoint_dir = args.checkpoint_dir
a_root = config['swb_audio_root']
t_root = config['swb_transcription_root']
expand_channel_dim = config['expand_channel_dim']
torch_device = args.torch_device
num_workers = int(args.num_workers)
dropout_rate = float(args.dropout_rate)

collate_fn=partial(audio_utils.pad_sequences_with_labels,
                        expand_channel_dim=expand_channel_dim)

##################################################################
####################  Setup Training Model  ######################
##################################################################

def load_noise_files():
    noise_files = librosa.util.find_files('/mnt/data0/jrgillick/projects/laughter-detection/data/extra_sounds')
    music_files = librosa.util.find_files('/mnt/data0/jrgillick/projects/laughter-detection/data/spotifyClips/clips/')
    noise_files += list(np.random.choice(music_files, 50))
    noise_signals = audio_utils.parallel_load_audio_batch(noise_files,n_processes=8,sr=8000)
    noise_signals = [s for s in noise_signals if len(s) > 8000]
    return noise_signals

def run_training_loop(n_epochs, model, device, checkpoint_dir,
    optimizer, iterator, log_frequency=25, val_iterator=None,
    gradient_clip=1., verbose=True):

    for epoch in range(n_epochs):
        start_time = time.time()

        # Run with Generator
        train_loss = run_epoch(model, 'train', device, iterator,
            checkpoint_dir=checkpoint_dir, optimizer=optimizer,
            log_frequency=log_frequency, checkpoint_frequency=log_frequency,
            clip=gradient_clip, val_iterator=val_iterator, verbose=verbose)

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
                val_loss, val_acc = _eval_batch(model, device, val_batch)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            except StopIteration:
                val_itr = iter(val_iterator)
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
            loss = criterion(output, trg)
            preds = torch.round(output)
            acc = torch.sum(preds==trg).float()/len(trg) #sum(preds==trg).float()/len(preds)
            return loss.item(), acc.item()

    def _train_batch(model,device,batch,batch_index=None,clip=None):

        if batch is None:
            print("None Batch")
            return 0.

        seqs, labs = batch


        src = torch.from_numpy(np.array(seqs)).float().to(device)
        trg = torch.from_numpy(np.array(labs)).float().to(device)

        optimizer.zero_grad()

        output = model(src).squeeze()
        
        criterion = nn.BCELoss()
        loss = criterion(output, trg)
        preds = torch.round(output)
        acc = torch.sum(preds==trg).float()/len(trg)
        
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        return loss.item(), acc.item()

    if not (bool(iterator) ^ bool(batches)):
        raise Exception("Must pass either `iterator` or batches")

    if mode.lower() not in ['train', 'eval']:
        raise Exception("`mode` must be 'train' or 'eval'")

    if mode.lower() == 'train' and validate_online:
        val_batches_per_epoch =  torch_utils.num_batches_per_epoch(val_iterator)
        val_batches_per_log = int(np.round(val_batches_per_epoch))#np.max(10,int(val_batches_per_epoch / log_frequency))
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

    writer = SummaryWriter(checkpoint_dir)

    if iterator is not None:
        batches_per_epoch = torch_utils.num_batches_per_epoch(iterator)
        batch_losses = []; batch_accs = []
        
        for i, batch in tqdm(enumerate(iterator)):
            # learning rate scheduling
            lr = (learning_rate - min_learning_rate)*decay_rate**(float(model.global_step))+min_learning_rate
            optimizer = optim.Adam(model.parameters(),lr=lr)
            
            batch_loss, batch_acc = _run_batch(model, device, batch, batch_index = i, clip=clip)
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

                writer.add_scalar('train/loss', train_loss_at_step, model.global_step)
                writer.add_scalar('train/acc', train_acc_at_step, model.global_step)
                writer.add_scalar('eval/loss', val_loss_at_step, model.global_step)
                writer.add_scalar('eval/acc', val_acc_at_step, model.global_step)
                batch_losses = []; batch_accs = [] # reset

            if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
                state = torch_utils.make_state_dict(model, optimizer, model.epoch,
                                    model.global_step, model.best_val_loss)
                torch_utils.save_checkpoint(state, is_best=True, checkpoint=checkpoint_dir)

            epoch_loss += batch_loss
            model.global_step += 1

        model.epoch += 1
        return epoch_loss / len(iterator)


##################################################################
####################  Set up Model Training  #####################
##################################################################

    
print("Initializing model...")
device = torch.device(torch_device if torch.cuda.is_available() else 'cpu')
print("Using device", device)
model = config['model'](dropout_rate=dropout_rate, linear_layer_size=config['linear_layer_size'])
model.set_device(device)
#model = model.to(device)
torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)
optimizer = optim.Adam(model.parameters())

if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir+'/last.pth.tar', model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    print("Beginning training...")


#########################################################
############   Do this once, keep in memory  ############
#########################################################

if augment_fn is not None:
    print("Loading background noise files...")
    noise_signals = load_noise_files()
    augment_fn = partial(augment_fn, noise_signals=noise_signals)
    augmented_feature_fn = partial(feature_fn, augment_fn=augment_fn)
else:
    augmented_feature_fn = feature_fn

print("Loading switchboard audio files...")
t0 = time.time()
with open(swb_train_audio_pkl_path, "rb") as f: # Loads all switchboard audio files
    h = pickle.load(f)
        
all_audio_files = librosa.util.find_files(a_root,ext='sph')
train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(t_root)
t_files_a, a_files = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
t_files_b, _ = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

def get_audios_from_text_data(data_file, h, sr=8000):
    audios = []
    df = pd.read_csv(data_file,sep='\t',header=None,
        names=['offset','duration','audio_path','label'])
    audio_paths = list(df.audio_path)
    offsets = list(df.offset)
    durations = list(df.duration)
    for i in tqdm(range(len(audio_paths))):
        aud = h[audio_paths[i]][int(offsets[i]*sr):int((offsets[i]+durations[i])*sr)]
        audios.append(aud)
    return audios

def make_text_dataset(t_files_a, t_files_b, audio_files,num_passes=1,n_processes=8):
    big_list = []
    assert(len(t_files_a)==len(t_files_b) and len(t_files_a)==len(audio_files))
    for p in range(num_passes):
        lines_per_file = Parallel(n_jobs=n_processes)(
            delayed(dataset_utils.get_laughter_speech_text_lines)(t_files_a[i],
                    t_files_b[i], audio_files[i]) for i in tqdm(range(len(t_files_a))))
        big_list += audio_utils.combine_list_of_lists(lines_per_file)
    return big_list

##################################################################
####################  Load Validation Data  ######################
##################################################################

with open(swb_val_audio_pkl_path, 'rb') as f:
    val_h = pickle.load(f)
val_audios = get_audios_from_text_data(val_data_text_path, val_h)

val_dataset = data_loaders.SwitchBoardLaughterDataset(
    data_file=val_data_text_path,
    audio_files = val_audios,
    feature_fn=feature_fn,
    batch_size=batch_size,
    sr=8000)

val_generator = torch.utils.data.DataLoader(
    val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn)

##################################################################
#######################  Run Training Loop  ######################
##################################################################

#for e in range(200):
while model.global_step < 200000:
    t0 = time.time()
    print("Preparing training set...")
    
    lines = make_text_dataset(t_files_a, t_files_b, a_files,num_passes=1)                      
    with open(train_data_text_path, 'w')  as f:
        f.write('\n'.join(lines))
    train_audios = get_audios_from_text_data(train_data_text_path, h)

    train_dataset = data_loaders.SwitchBoardLaughterDataset(
        data_file=train_data_text_path,
        audio_files = train_audios,
        feature_fn=augmented_feature_fn,
        batch_size=batch_size,
        sr=8000)

    training_generator = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
    
    run_training_loop(n_epochs=1, model=model, device=device, iterator=training_generator,
        checkpoint_dir=checkpoint_dir, optimizer=optimizer, log_frequency=log_frequency,
        val_iterator=val_generator,verbose=True)
    

