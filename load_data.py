from torch.utils.data import DataLoader
from lhotse import CutSet, Fbank, FbankConfig, MonoCut, LilcomFilesWriter
from lhotse.dataset import VadDataset, SingleCutSampler
from lhotse.recipes import prepare_icsi
from lhotse import SupervisionSegment, SupervisionSet, RecordingSet
from lad import LadDataset
import pandas as pd
import pickle
import os

DEBUG = False
FORCE_MANIFEST_RELOAD = False # allows overwriting already stored manifests
FORCE_FEATURE_RECOMPUTE = True # allows overwriting already computed features

SPLITS = ['train', 'dev', 'test']


def create_dataloader():
    if DEBUG:
        data_dir = 'data/icsi/'
        # lhotse_dir: Directory which will contain manifest and cutset dumps from lhotse
        lhotse_dir = os.path.join(data_dir, 'test')
        audio_dir = 'data/test_speech/'
        transcripts_dir = 'data/test_transcripts/'
        manifest_dir = os.path.join(lhotse_dir, 'manifests')
        dataframe = 'dummy_df.csv'
        feats_path = os.path.join(lhotse_dir, 'feats')
        cuts_file = os.path.join(lhotse_dir, 'debug_cuts.jsonl')
        cutset_dir = os.path.join(lhotse_dir, 'cutsets')
        print('IN DEBUG MODE - loading small amount of data')
    else:
        data_dir = 'data/icsi/'
        # lhotse_dir: Directory which will contain manifest and cutset dumps from lhotse
        lhotse_dir = os.path.join(data_dir, 'lhotse')
        audio_dir = 'data/icsi/speech/'
        transcripts_dir = 'data/icsi'
        manifest_dir = os.path.join(lhotse_dir, 'manifests')
        feats_path = os.path.join(lhotse_dir, 'feats')
        dataframe = 'val_df.csv'
        cuts_file = os.path.join(lhotse_dir, 'cuts_with_feats.jsonl')
        cutset_dir = os.path.join(lhotse_dir, 'cutsets')


    # Prepare data manifests from a raw corpus distribution.
    # The RecordingSet describes the metadata about audio recordings;
    # the sampling rate, number of channels, duration, etc.
    # The SupervisionSet describes metadata about supervision segments:
    # the transcript, speaker, language, and so on.
    if(os.path.isdir(manifest_dir) and not FORCE_MANIFEST_RELOAD):
        print("LOADING MANIFEST DIR FROM DISK - not from raw icsi files")
        icsi = {'train': {}, 'dev': {}, 'test': {}}
        for split in ['train', 'dev', 'test']:
            rec_set = RecordingSet.from_jsonl(os.path.join(
                manifest_dir, f'recordings_{split}.jsonl'))
            sup_set = SupervisionSet.from_jsonl(os.path.join(
                manifest_dir, f'supervisions_{split}.jsonl'))
            icsi[split]['recordings'] = rec_set
            icsi[split]['supervisions'] = sup_set
    else:
        icsi = prepare_icsi(
            audio_dir=audio_dir, transcripts_dir=transcripts_dir, output_dir=manifest_dir)

    # Load the channel to id mapping from disk
    # If this changed at some point (which it shouldn't) this file would have to
    # be recreated
    # TODO: find a cleaner way to implement this
    chan_map_file = open(os.path.join(data_dir, 'chan_idx_map.pkl'), 'rb')
    chan_idx_map = pickle.load(chan_map_file)

    # Read data_dfs containing the samples for train,val,test split
    dfs = {}
    for split in SPLITS:
        dfs[split] = pd.read_csv(os.path.join(data_dir, 'data_dfs', f'{split}_df.csv'))

    # CutSet is the workhorse of Lhotse, allowing for flexible data manipulation.
    # We use the existing dataframe to create a corresponding cut for each row
    # Supervisions stating laugh/non-laugh are attached to each cut
    # No audio data is actually loaded into memory or stored to disk at this point.
    # Columns of dataframes look like this:
    #   cols = ['start', 'duration', 'sub_start', 'sub_duration', 'audio_path', 'label']

    cutset_dict = {} # will contain CutSets for different splits
    for split, df in dfs.items():
        cut_list = []
        for ind, row in df.iterrows():
            meeting_id = row.audio_path.split('/')[0]
            channel = row.audio_path.split('/')[1].split('.')[0]
            chan_id = chan_idx_map[meeting_id][channel]
            if DEBUG:
                # The meeting used in dummy_df is in the train-split
                rec = icsi['train']['recordings'][meeting_id]
            else:
                # In the icsi recipe the validation split is called 'dev' split
                rec = icsi[split]['recordings'][meeting_id]
            # Create supervision segment indicating laughter or non-laughter by passing a
            # dict to the custom field -> {'is_laugh': 0/1}
            sup = SupervisionSegment(id=f'sup_{split}_{ind}', recording_id=rec.id, start=row.sub_start,
                                    duration=row.sub_duration, channel=chan_id, custom={'is_laugh': row.label})
            cut = MonoCut(id=f'{split}_{ind}', start=row.sub_start, duration=row.sub_duration,
                        recording=rec, channel=chan_id, supervisions=[sup])
            cut_list.append(cut)

        cutset_dict[split] = CutSet.from_cuts(cut_list)

    print('Write cutset_dict to disk...')
    with open(os.path.join(cutset_dir, 'cutset_dict_without_feats.pkl'), 'wb') as f:
        pickle.dump(cutset_dict, f)
    
    for split, cutset in cutset_dict.items():
        print('Computing features for {split}...')
        # Choose frame_shift value to match the hop_length of Gillick et al
        # 0.2275 = 16 000 / 364 -> [frame_rate / hop_length]
        f2 = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))

        if(os.path.isfile(cuts_file) and not FORCE_FEATURE_RECOMPUTE):
            print("LOADING FEATURES FROM DISK - NOT RECOMPUTING")
            cuts = CutSet.from_jsonl(cuts_file)
        else:
            cuts = cutset.compute_and_store_features(
                extractor=f2,
                storage_path=feats_path,
                num_jobs=1,
                storage_type=LilcomFilesWriter
            )
            cuts = cuts.shuffle()
            cuts.to_jsonl(os.path.join(cutset_dir, f'{split}_cutset_with_feats.jsonl'))

    # Shuffle cutset for better training. In the data_dfs the rows aren't shuffled.
    # At the top are all speech rows and the bottom all laugh rows
    # Construct a Pytorch Dataset class for Voice Activity Detection task:
    dataset = LadDataset()
    sampler = SingleCutSampler(cuts, max_cuts=32)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
    return dataloader


if __name__ == '__main__':
    create_dataloader()
