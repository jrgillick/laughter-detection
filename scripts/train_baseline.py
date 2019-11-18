import os, sys, pickle, time, librosa, torch, numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
import dataset_utils, audio_utils, data_loaders, torch_utils
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    
from tensorboardX import SummaryWriter
batch_size=500
collate_fn = audio_utils.pad_sequences_with_labels



##################################################################
####################  Load Validation Data  ######################
##################################################################

with open("/data/jrgillick/projects/laughter-detection/data/switchboard/val/val_1.pkl", "rb") as f:
    val_audios = pickle.load(f)

val_dataset = data_loaders.SwitchBoardLaughterDataset(
    data_file='/data/jrgillick/projects/laughter-detection/data/switchboard/val/val_1.txt',
    audio_files = val_audios,
    feature_fn=audio_utils.featurize_mfcc,
    batch_size=batch_size,
    sr=8000)

val_generator = torch.utils.data.DataLoader(
    val_dataset, num_workers=8, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn)

##################################################################
####################  Setup Training Model  ######################
##################################################################

def run_training_loop(n_epochs, model, device, checkpoint_dir,
    optimizer, iterator, val_iterator=None,
    gradient_clip=1., verbose=True):

    for epoch in range(n_epochs):
        start_time = time.time()

        # Run with Generator
        train_loss = run_epoch(model, 'train', device, iterator,
            optimizer=optimizer, clip=gradient_clip,
            val_iterator=val_iterator, checkpoint_dir=checkpoint_dir,
            verbose=verbose)

        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = torch_utils.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            
def run_epoch(model, mode, device, iterator, optimizer=None, clip=None,
                batches=None, log_frequency=25, checkpoint_frequency=25,
                validate_online=True, val_iterator=None, val_batches=None,
                checkpoint_dir=None, verbose=True):

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
        
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        return loss.item()

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
        batch_losses = []
        
        for i, batch in tqdm(enumerate(iterator)):
            batch_loss = _run_batch(model, device, batch, batch_index = i, clip=clip)
            batch_losses.append(batch_loss)

            if log_frequency is not None and (model.global_step + 1) % log_frequency == 0:
                val_itr, val_loss_at_step, val_acc_at_step = _eval_for_logging(model, device,
                    val_itr, val_iterator, val_batches_per_log)

                is_best = (val_loss_at_step < model.best_val_loss)
                if is_best:
                    model.best_val_loss = val_loss_at_step

                train_loss_at_step = np.mean(batch_losses)

                if verbose:
                    print("\nLogging at step: ", model.global_step)
                    print("Train loss: ", train_loss_at_step)
                    print("Val loss: ", val_loss_at_step)
                    print("Val Accuracy: ", val_acc_at_step)

                writer.add_scalar('train/loss', train_loss_at_step, model.global_step)
                writer.add_scalar('eval/loss', val_loss_at_step, model.global_step)
                writer.add_scalar('eval/acc', val_acc_at_step, model.global_step)
                batch_losses = [] # reset

            if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
                state = torch_utils.make_state_dict(model, optimizer, model.epoch,
                                    model.global_step, model.best_val_loss)
                torch_utils.save_checkpoint(state, is_best=True, checkpoint=checkpoint_dir)

            epoch_loss += batch_loss
            model.global_step += 1

        model.epoch += 1
        return epoch_loss / len(iterator)
    
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MLPModel(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hid_dim1)
        self.linear2 = nn.Linear(hid_dim1, hid_dim2)
        self.linear3 = nn.Linear(hid_dim2, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hid_dim1)
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim2)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf
        
    def forward(self, src):
        src = src.view((-1,76*40))
        hidden1 = self.linear1(src)
        hidden1 = self.bn1(hidden1)
        hidden1 = self.dropout(hidden1)
        hidden1 = F.relu(hidden1)
        
        hidden2 = self.linear2(hidden1)
        hidden2 = self.bn2(hidden2)
        hidden2 = self.dropout(hidden2)
        hidden2 = F.relu(hidden2)
        output = self.linear3(hidden2)
        output = torch.sigmoid(output)
        return output
    
print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_model = MLPModel(input_dim=76*40,hid_dim1=600,hid_dim2=100,dropout=0.5).to(device)
model = mlp_model
torch_utils.count_parameters(model)
model.apply(torch_utils.init_weights)
optimizer = optim.Adam(model.parameters())

checkpoint_dir = '/data/jrgillick/projects/laughter-detection/checkpoints/mlp_mfcc'
if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir+'/last.pth.tar', model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    print("Beginning training...")

#cmd = "python make_switchboard_text_dataset.py --output_txt_file=/data/jrgillick/projects/laughter-detection/data/switchboard/train/train_1.txt --switchboard_audio_path=/data/corpora/switchboard-1/97S62/ --switchboard_transcriptions_path=/data/corpora/switchboard-1/swb_ms98_transcriptions/ --data_partition=train --load_audio_path=/data0/project/microtuning/misc/swb_train_audios.pkl"

#########################################################
############   Do this once, keep in memory  ############
#########################################################
load_audio_path = '/data0/project/microtuning/misc/swb_train_audios.pkl'
output_txt_file = '/data/jrgillick/projects/laughter-detection/data/switchboard/train/train_1.txt'
a_root = '/data/corpora/switchboard-1/97S62/'
t_root = '/data/corpora/switchboard-1/swb_ms98_transcriptions/'

print("Loading switchboard audio files...")
t0 = time.time()
with open(load_audio_path, "rb") as f: # Loads all switchboard audio files
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
    try:
        big_list = []
        assert(len(t_files_a)==len(t_files_b) and len(t_files_a)==len(audio_files))
        for p in range(num_passes):
            lines_per_file = Parallel(n_jobs=n_processes)(
                delayed(dataset_utils.get_laughter_speech_text_lines)(t_files_a[i],
                        t_files_b[i], audio_files[i]) for i in tqdm(range(len(t_files_a))))
            big_list += audio_utils.combine_list_of_lists(lines_per_file)
        return big_list
    except:
        print("Error making dataset. Retrying...")
        return make_text_dataset(t_files_a, t_files_b, audio_files, num_passes, n_processes)
#########################################################
############   Do this once, keep in memory  ############
#########################################################

for e in range(200):
    t0 = time.time()
    print("Preparing training set...")
    
    
    lines = make_text_dataset(t_files_a, t_files_b, a_files,num_passes=1)                      
    with open(output_txt_file, 'w')  as f:
        f.write('\n'.join(lines))
    audios = get_audios_from_text_data(output_txt_file, h)
    #os.system(cmd)
    #print("Created training set in ", time.time()-t0, "seconds")
    #with open("/data/jrgillick/projects/laughter-detection/data/switchboard/train/train_1.pkl", "rb") as f:
    #    audios = pickle.load(f)

    train_dataset = data_loaders.SwitchBoardLaughterDataset(
        data_file=output_txt_file,
        audio_files = audios,
        feature_fn=audio_utils.featurize_mfcc,
        batch_size=batch_size,
        sr=8000)

    training_generator = torch.utils.data.DataLoader(
        train_dataset, num_workers=8, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
    
    run_training_loop(n_epochs=1, model=model, device=device, iterator=training_generator,
        checkpoint_dir=checkpoint_dir, optimizer=optimizer,
        val_iterator=val_generator,verbose=True)
    

