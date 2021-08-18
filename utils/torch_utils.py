import numpy as np, os, sys, shutil, time, math
import torch
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import text_utils

# Import different progress bar depending on environment
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
if 'ipykernel' in sys.modules:
	from tqdm import tqdm_notebook as tqdm
else:
	from tqdm import tqdm


##################### INITIALIZATION ##########################

def count_parameters(model):
	counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'The model has {counts:,} trainable parameters')

def init_weights(model):
	for name, param in model.named_parameters():
		nn.init.normal_(param.data, mean=0, std=0.01)


##################### TENSOR OPERATIONS #######################

def torch_one_hot(y, device, n_dims=None):
	""" Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
	# Source https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
	y_tensor = y.data if isinstance(y, Variable) else y
	y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
	n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
	y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
	y_one_hot = y_one_hot.view(*y.shape, -1)
	outp = Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
	return outp.to(device)

def create_embedding_layer(weight_matrix, trainable=True):
	vocab_size, embedding_dim = weight_matrix.shape
	embedding_layer = nn.Embedding(vocab_size, embedding_dim)
	embedding_layer.load_state_dict({'weight': torch.from_numpy(weight_matrix)})
	if not trainable:
		embedding_layer.weight.requires_grad = False
	return embedding_layer

def num_batches_per_epoch(generator):
	return len(generator.dataset)/generator.batch_size

def compute_bow_loss(output, trg, device):
	# output shape: (seq_len, batch_size, output_dim). eg (75, 32, 64)
	# trg shape: (seq_len, batch_size). eg (75, 32)

	output_dim = output.shape[2]
	batch_size = output.shape[1]
	seq_len = output.shape[0]

	total_loss = torch.zeros(1)[0].to(device)

	for i in range(batch_size):
		single_output = output[:,i,:].argmax(1) # predicted bag of words
		bow_output_tensor = torch.zeros(output_dim).to(device)
		for ind in single_output: bow_output_tensor[ind] = 1
		single_target = trg[:,i] # target bag of words
		bow_target_tensor = torch.zeros(output_dim).to(device)
		for ind in single_target: bow_target_tensor[ind] = 1
		single_loss = (bow_target_tensor - bow_output_tensor).abs().sum()
		total_loss += single_loss

	return total_loss / batch_size / seq_len


################## CHECKPOINTING ##############################

def save_checkpoint(state, is_best, checkpoint):
	"""Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
	checkpoint + 'best.pth.tar'
	Args:
		state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
		is_best: (bool) True if it is the best model seen till now
		checkpoint: (string) folder where parameters are to be saved

	Modified from: https://github.com/cs230-stanford/cs230-code-examples/
	"""
	filepath = os.path.join(checkpoint, 'last.pth.tar')
	if not os.path.exists(checkpoint):
		print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
		os.mkdir(checkpoint)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
	"""Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
	optimizer assuming it is present in checkpoint.
	Args:
		checkpoint: (string) filename which needs to be loaded
		model: (torch.nn.Module) model for which the parameters are loaded
		optimizer: (torch.optim) optional: resume optimizer from checkpoint

	Modified from: https://github.com/cs230-stanford/cs230-code-examples/
	"""
	if not os.path.exists(checkpoint):
		raise ("File doesn't exist {}".format(checkpoint))
	else:
		print("Loading checkpoint at:", checkpoint)
	checkpoint = torch.load(checkpoint)
	model.load_state_dict(checkpoint['state_dict'])

	if optimizer:
		optimizer.load_state_dict(checkpoint['optim_dict'])

	if 'epoch' in checkpoint:
		model.epoch = checkpoint['epoch']

	if 'global_step' in checkpoint:
		model.global_step = checkpoint['global_step'] + 1
		print("Loading checkpoint at step: ", model.global_step)

	if 'best_val_loss' in checkpoint:
		model.best_val_loss = checkpoint['best_val_loss']

	return checkpoint

def make_state_dict(model, optimizer=None, epoch=None, global_step=None,
	best_val_loss=None):
	return {'epoch': epoch, 'global_step': global_step,
		'best_val_loss': best_val_loss, 'state_dict': model.state_dict(),
		'optim_dict' : optimizer.state_dict()
	}



##################### TRAINING METHODS ######################

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def run_training_loop(n_epochs, model, device, loss_type, checkpoint_dir,
	optimizer, iterator, teacher_forcing_ratio=0.5, val_iterator=None,
	gradient_clip=1., verbose=True):

	for epoch in range(n_epochs):
		start_time = time.time()

		# Run with Generator
		train_loss = run_epoch(model, 'train', device, loss_type=loss_type,
			optimizer=optimizer, clip=gradient_clip, iterator=iterator,
			teacher_forcing_ratio=teacher_forcing_ratio,
			val_iterator=val_iterator, checkpoint_dir=checkpoint_dir,
			verbose=verbose)

		if verbose:
			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

def run_epoch(model, mode, device, loss_type='x_ent', optimizer=None, clip=1,
				  iterator=None, batches=None, teacher_forcing_ratio=0.5,
				  log_frequency=100, checkpoint_frequency=100, validate_online=True,
				  val_iterator=None, val_batches=None, checkpoint_dir=None,
				  verbose=True):

	""" args:
			mode: 'train' or 'eval'
			loss_type: 'x_ent' for nn.CrossEntropyLoss, 'ctc' for nn.CTCLoss
	"""

	def _get_ctc_loss(output, trg, device):
		ctc_batch_size = output.shape[1]
		ctc_input_length = output.shape[0]
		ctc_inputs=output.log_softmax(2)
		ctc_trgs = trg.permute((1,0))
		ctc_input_lengths = torch.full(size=(ctc_batch_size,),
			fill_value=ctc_input_length, dtype=torch.long)

		# TODO figure this part out better..
		#ctc_trg_lengths = torch.full(size=(ctc_batch_size,), fill_value=ctc_input_length, dtype=torch.long)
		ctc_trg_list = []
		for i in range(ctc_batch_size):
			nz = (ctc_trgs[i,:]==0).nonzero()
			if len(nz) == 0 or nz[0] > ctc_input_length/2:
				ctc_trg_list.append(torch.tensor([int(ctc_input_length/2)]).to(device))
				#ctc_trg_list.append(ctc_input_lengths[0:1].to(device))
			else:
				ctc_trg_list.append(nz[0].detach())

		try:
			ctc_trg_lengths = torch.cat(ctc_trg_list).to(device)
		except:
			import pdb; pdb.set_trace()


		loss_ctc = nn.CTCLoss()(ctc_inputs, ctc_trgs, ctc_input_lengths, ctc_trg_lengths)

		loss = loss_ctc
		if torch.isinf(loss):
			import pdb; pdb.set_trace()
		return loss

	def _eval_for_logging(model, device, val_itr, val_iterator,
		val_batches_per_log, loss_type):
		model.eval()
		val_losses = []
		for j in range(val_batches_per_log):
			try:
				val_batch = val_itr.next()
				val_losses.append(_eval_batch(model, device, val_batch, loss_type, teacher_forcing_ratio=0))
			except StopIteration:
				val_itr = iter(val_iterator)
		model.train()
		return val_itr, np.mean(val_losses)

	def _eval_batch(model, device, batch, loss_type, teacher_forcing_ratio=0.,
					batch_index=None, accumulation_steps=16):
		if batch is None:
			print("None Batch")
			return 0.

		with torch.no_grad():
			seqs, labs = batch

			input_tensor = torch.from_numpy(np.array(seqs)).float().permute(
				(1,0,2)).to(device)
			target_tensor = torch.from_numpy(np.array(labs)).permute(
				(1,0)).to(device)

			src = input_tensor
			trg = target_tensor # [trg sent len, batch size]

			#NO teacher forcing
			output = model(src, trg, 0) #[trg sent len, batch size, output dim]

			if loss_type == 'ctc':
				loss = _get_ctc_loss(output, trg, device)
				return loss.item()

			else:
				# cross entropy

				# dim: [(trg sent len - 1) * batch size, output dim]
				output = output[1:].view(-1, output.shape[-1])

				# dim: [(trg sent len - 1) * batch size]
				trg = trg[1:].contiguous().view(-1)

				c = nn.CrossEntropyLoss(ignore_index = 0) #output_vocab[text_utils.PAD_SYMBOL]
				loss = c(output, trg) #/ accumulation_steps
				return loss.item()

	def _train_batch(model, device, batch, loss_type, teacher_forcing_ratio=0.5,
					 batch_index=None, accumulation_steps=16):

		if batch is None:
			print("None Batch")
			return 0.

		seqs, labs = batch

		input_tensor = torch.from_numpy(np.array(seqs)).float().permute(
			(1,0,2)).to(device)
		target_tensor = torch.from_numpy(np.array(labs)).permute(
			(1,0)).to(device)

		src = input_tensor
		trg = target_tensor

		optimizer.zero_grad()

		output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

		if loss_type == 'ctc':
			loss = _get_ctc_loss(output, trg, device)
			#bow_loss = compute_bow_loss(output, trg.permute((1,0)))

		else:
			flat_output = output[1:].view(-1, output.shape[-1])
			flat_trg = trg[1:].contiguous().view(-1)
			c = nn.CrossEntropyLoss(ignore_index = 0) #output_vocab[text_utils.PAD_SYMBOL]
			loss = c(flat_output, flat_trg)

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()

		return loss.item()

	if not (bool(iterator) ^ bool(batches)):
		raise Exception("Must pass either `iterator` or batches")

	if mode.lower() not in ['train', 'eval']:
		raise Exception("`mode` must be 'train' or 'eval'")

	if mode.lower() == 'train' and validate_online:
		val_batches_per_epoch =  num_batches_per_epoch(val_iterator)
		val_batches_per_log = int(val_batches_per_epoch / log_frequency)
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
		batches_per_epoch = num_batches_per_epoch(iterator)
		batch_losses = []
		for i, batch in tqdm(enumerate(iterator)):
			batch_loss = _run_batch(model, device, batch, loss_type, teacher_forcing_ratio, batch_index = i)
			batch_losses.append(batch_loss)

			if log_frequency is not None and (model.global_step + 1) % log_frequency == 0:
				val_itr, val_loss_at_step = _eval_for_logging(model, device,
					val_itr, val_iterator, val_batches_per_log, loss_type)

				is_best = (val_loss_at_step < model.best_val_loss)
				if is_best:
					model.best_val_loss = val_loss_at_step

				train_loss_at_step = np.mean(batch_losses)

				train_ppl = math.exp(train_loss_at_step)
				val_ppl = math.exp(val_loss_at_step)

				if verbose:
					print("\nLogging at step: ", model.global_step)
					print("Train loss: ", train_loss_at_step, " | Train PPL: ", train_ppl)
					print("Val loss: ", val_loss_at_step, " | Val PPL: ", val_ppl)

				writer.add_scalar('train/loss', train_loss_at_step, model.global_step)
				writer.add_scalar('eval/loss', val_loss_at_step, model.global_step)
				writer.add_scalar('train/ppl', train_ppl, model.global_step)
				writer.add_scalar('eval/ppl', val_ppl, model.global_step)
				batch_losses = [] # reset

			if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
				state = make_state_dict(model, optimizer, model.epoch,
									model.global_step, model.best_val_loss)
				save_checkpoint(state, is_best=True, checkpoint=checkpoint_dir)

			epoch_loss += batch_loss
			model.global_step += 1

		model.epoch += 1
		return epoch_loss / len(iterator)
	else:
		# TODO this is out of date
		batches_per_epoch = len(batches)
		for batch_index in range(len(batches)):
			epoch_loss += _run_batch(model, device, batches[batch_index], loss_type, teacher_forcing_ratio)
		return epoch_loss / len(batches)



##################### PREDICTING FROM TRAINED MODELS #####################


class Predictor:
    def __init__(self, dataset, filepaths, model=None, reverse_vocab=None, generator=None,
                 label_paths=None, batch_size=None, labels=None, label_fn=None,
                collate_fn=None):

        # Validate args
        if generator is None and collate_fn is None:
            raise Exception("Need either `generator` or `collate_fn`")

        self.reverse_vocab = reverse_vocab
        self.batch_size = batch_size

        # Create a new Dataset of the same type as the given one
        # Copy over all the attributes except for the file paths
        d_vars = vars(dataset)

        self.dataset = type(dataset)(filepaths,
            feature_and_label_fn=d_vars['feature_and_label_fn'],
            feature_fn=d_vars['feature_fn'],
            label_paths=label_paths, labels=labels, label_fn=label_fn,
            does_subsample=d_vars['does_subsample'],
            **d_vars['kwargs'])

        self.model=model

        # Create a new Generator of the same type as the given one
        # And copy over the attributes if provided
        if generator is not None:
            g_vars = vars(generator)
            self.generator=type(generator)(self.dataset,
                num_workers=g_vars['num_workers'],
                batch_size=g_vars['batch_size'],
                collate_fn=g_vars['collate_fn'])
        else:
            self.generator = torch.utils.data.DataLoader(
            self.dataset, num_workers=0, batch_size=self.batch_size or len(filepaths),
            shuffle=False, collate_fn=collate_fn)

    def predict(self):
        to_return = []
        for i_batch, batch in enumerate(self.generator):
            seqs, labs = batch

            if labs is not None:
                # Convert to readable labels with reverse vocab
                labs = [text_utils.readable_outputs(
                    s, self.reverse_vocab) for s in np.array(labs)]

            # Run model forward
            with torch.no_grad():
                src = torch.from_numpy(np.array(seqs)).float().permute((1,0,2)).cpu()#.to(device)
                output = self.model(src).cpu() # No labels given

            # Remove batch dimension, get arxmax from one hot, and convert to numpy
            output_seqs = np.argmax(output.cpu().numpy(), axis=-1).T

            readable_preds = [text_utils.readable_outputs(
                s, self.reverse_vocab) for s in output_seqs]

            to_return.append( (readable_preds, labs) )
        return to_return

class OneFileDatasetPredictor:
    def __init__(self, dataset, index, model=None, reverse_vocab=None,
                 batch_size=None, generator=None,collate_fn=None):

        # Validate args
        if generator is None and collate_fn is None:
            raise Exception("Need either `generator` or `collate_fn`")

        self.reverse_vocab = reverse_vocab
        self.batch_size = batch_size or 1

        # Create a new Dataset of the same type as the given one
        # Copy over all the attributes except for the file paths
        d_vars = vars(dataset)

        self.dataset = type(dataset)(filepath=d_vars['filepath'],
            feature_and_label_fn=d_vars['feature_and_label_fn'],
            start_index = index, end_index = index+self.batch_size,
            load_fn=d_vars['load_fn'],
            **d_vars['kwargs'])

        self.model=model

        # Create a new Generator of the same type as the given one
        # And copy over the attributes if provided
        if generator is not None:
            g_vars = vars(generator)
            self.generator=type(generator)(self.dataset,
                num_workers=g_vars['num_workers'],
                batch_size=g_vars['batch_size'],
                collate_fn=g_vars['collate_fn'])
        else:
            self.generator = torch.utils.data.DataLoader(
            self.dataset, num_workers=0, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn)

    def predict(self):
        to_return = []
        for i_batch, batch in enumerate(self.generator):
            seqs, labs = batch

            if labs is not None:
                # Convert to readable labels with reverse vocab
                labs = [text_utils.readable_outputs(
                    s, self.reverse_vocab) for s in np.array(labs)]

            # Run model forward
            with torch.no_grad():
                src = torch.from_numpy(np.array(seqs)).float().permute((1,0,2))#.to(device)
                output = self.model(src) # No labels given

            # Remove batch dimension, get arxmax from one hot, and convert to numpy
            output_seqs = np.argmax(output.cpu().numpy(), axis=-1).T

            readable_preds = [text_utils.readable_outputs(
                s, self.reverse_vocab) for s in output_seqs]

            to_return.append( (readable_preds, labs) )
        return to_return