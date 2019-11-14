from __future__ import unicode_literals
import os, sox, numpy as np, time
import youtube_dl
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

# Define the tags corresponding to laughter types in AudioSet
laugh_id_dict = {}
laugh_id_dict['/m/01j3sz'] = "Laughter"
laugh_id_dict['/t/dd00001'] = "Baby laughter"
laugh_id_dict['/m/07r660_'] = "Giggle"
laugh_id_dict['/m/07s04w4'] = "Snicker"
laugh_id_dict['/m/07sq110'] = "Belly laugh"
laugh_id_dict['/m/07rgt08'] = "Chuckle, chortle"
laugh_keys = list(laugh_id_dict.keys())

def laugh_id_multihot(laugh_type_list):
	""" Mapping from list of ID strings to multihot vector
	"""
	v = np.zeros(len(laugh_keys))
	for i, k in enumerate(laugh_keys):
		if k in laugh_type_list:
			v[i] = 1.
	return v

def get_laughter_infolist(csv_file, mode):
	"""Get a list of laughter/no-laughter files to download from youtube.


	Args:
		csv_file: One of the metadata csv files from AudioSet.  This contains
			youtube ID's, start and end times, and tags that indicated which
			types of audio events occur in each 10 second clip.
		mode: One of 'positive', 'negative', or 'both'. 'positive' will download
			positive examples (clips with laughter), 'negative' will return a
			list of clips without laughter, and both will give both.

	Returns:
		A list of dicts. Each dict contains yt_id, start_time, end_time, and
		tag_strings (a list of audio set tags). tag_strings corresponds to the
		keys for looking up AudioSet tags - e.g. ['/m/01j3sz', '/t/dd00001']

	"""

	# Read the csv file from AudioSet and remove headers and blank lines.
	lines = open(csv_file).read().split('\n')[3:-1]

	laughter_lines = []
	all_non_laughter_lines = []

	# Separate each entry in one list of clips with laughter, and one without.
	for l in lines:
		yt_id, start_time, end_time, tag_strings = l.split(', ')
		tag_strings = tag_strings.replace('"','').split(',')
		laugh_tags = [laugh_id_dict[t] for t in tag_strings if t in laugh_keys]
		if laugh_tags:
			laughter_lines.append({'yt_id': yt_id, 'start_time': start_time,
				'end_time': end_time, 'tag_strings': tag_strings})
		else:
			all_non_laughter_lines.append({'yt_id': yt_id, 'start_time': start_time,
				'end_time': end_time, 'tag_strings': tag_strings})

	# Sample an equal number of non-laughter clips to balance the data.
	n_examples = len(laughter_lines)
	np.random.seed(0)
	non_laughter_lines = list(np.random.choice(all_non_laughter_lines, n_examples, replace=False))

	if mode == 'positive':
		return laughter_lines
	elif mode == 'negative':
		return non_laughter_lines
	elif mode == 'both':
		return laughter_lines + non_laughter_lines
	else:
		raise Exception("invalid input")

def download_laughter_audio_file(laughter_info, destination_dir):
	""" Download an audio file from YT, trim to the relevant segment, and save.

	File will be downloaded from YT, converted to .mp3, clipped to just the
		relevant 10s specified by 'start_time' and 'end_time', and then saved
		into destination_dir. It will be saved in destination_dir with a name
                following the pattern yt_<id>.mp3.

	Args:
		laughter_info: A dict of the form created in get_laughter_infolist().
			Keys are 'yt_id', 'start_time', 'end_time', and 'tag_strings'.
		destination_dir: Path to the directory to store downloaded audio.

	Returns: None
	"""

	# Define some paths to use later.
	mp3_file = "yt_" + laughter_info['yt_id'] + ".mp3"
	destination_file = os.path.join(destination_dir, mp3_file)

	# Skip if the file already exists in the desination directory.
	if os.path.exists(destination_file):
		return

	# Create a temp directory with a random name to operate in.
	tmpdir = 'tmp_' + str(np.random.randint(9999999)) + '/'
	# Make sure tmpdir names don't collide during parallel processing.
	while os.path.exists(tmpdir):
		tmpdir = 'tmp_' + str(np.random.randint(9999999)) + '/'
	os.mkdir(tmpdir)
	os.chdir(tmpdir)

	yt_url = "http://www.youtube.com/watch?v=" + laughter_info['yt_id']
	# Could fail if the video or YT accound has been taken down.
	try:
		# Download into the current working dir, which has been set to tmpdir.
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                     ydl.download([yt_url])
		downloaded_file = os.listdir()[0]
		tmp_mp3_file = "yt_" + laughter_info['yt_id'] + "tmp.mp3"
		# Rename to avoid any weird characters or spaces.
		os.rename(downloaded_file, tmp_mp3_file)
		# Clip to the relevant 10 seconds using sox
		start_time = float(laughter_info['start_time'])
		end_time = float(laughter_info['end_time'])
		tfm = sox.Transformer()
		tfm.trim(start_time, end_time)
		tfm.build(tmp_mp3_file, mp3_file)
		os.remove(tmp_mp3_file)
		# back out of tmpdir
		os.chdir('..')
		# move file to final destination
		os.rename(tmpdir+mp3_file, destination_file)
		# remove tmpdir
		os.rmdir(tmpdir)
	except:
		# If there's an error, back out to original dir and remove tmpdir
		print(yt_url + " failed.")
		os.chdir('..')
		os.rmdir(tmpdir)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-o','--output_dir', help='output folder path', required=True)
	parser.add_argument('-i','--input_csv', help='input csv path', required=True)
	parser.add_argument('-n','--num_processes', help='number of parallel processes', required=True)
	# Set mode to download positive examples, negative examples, or both
	parser.add_argument('-m','--mode', help='{positive, negative, both}', required=True)

	args = vars(parser.parse_args())
	output_dir = args["output_dir"]
	input_csv = args["input_csv"]
	num_processes = int(args["num_processes"])
	mode = args["mode"]

	print("Reading metadata...")
	laughter_lines = get_laughter_infolist(input_csv, mode)
	print("Downloading %d clips..." % (len(laughter_lines)))

	t0 = time.time()

	if num_processes>1:
		Parallel(n_jobs=num_processes)(delayed(download_laughter_audio_file)(laughter_info, output_dir) for laughter_info in tqdm(laughter_lines))
	else:
		for laughter_info in tqdm(laughter_lines):
			download_laughter_audio_file(laughter_info, output_dir)

	print("Finished in %f seconds" % (time.time()-t0))




