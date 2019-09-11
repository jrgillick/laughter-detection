import os
import sys
import laugh_segmenter
import tgt

def seconds_to_frames(s):
	return(int(s*100))

def parse_inputs():
	process = True

	try:
		a_file = sys.argv[1]
	except:
		print("Enter the audio file path as the first argument")
		process = False

	try:
		model_path = sys.argv[2]
	except:
		print("Enter the stored model path as the second argument")
		process = False

	try:
		output_audio_path = sys.argv[3]
	except:
		print("Enter the output audio path as the third argument")
		process = False

	try:
		threshold = float(sys.argv[4])
	except:
		threshold= 0.5

	try:
		min_length = float(sys.argv[5])
	except:
		min_length = 0.2

	try:
		save_to_textgrid = sys.argv[6] == 'True'
	except:
		save_to_textgrid = False

	if process:
		return (a_file, model_path, output_audio_path, threshold, min_length, save_cuts)
	else:
		return False

# Usage: python segment_laughter.py <input_audio_file> <stored_model_path> <output_folder> <save_to_textgrid>

if __name__ == '__main__':
	if parse_inputs():
		input_path, model_path, output_path, threshold, min_length, save_to_textgrid = parse_inputs()
		min_length = seconds_to_frames(min_length)

		laughs = laugh_segmenter.segment_laughs(input_path, model_path, output_path,
                                                        threshold, min_length, save_to_textgrid) 
		print(); print("found %d laughs." % (len (laughs)))

		if not save_to_textgrid:
			for laugh in laughs:
				print(laugh)
		else:
			tg = tgt.TextGrid()
			laughs_tier = tgt.IntervalTier(name='laughter', objects=[
				tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
			tg.add_tier(laughs_tier)
			fname = os.path.splitext(os.path.basename(input_path))[0]
			tgt.write_to_file(tg, os.path.join(output_path, fname + '_laughter.TextGrid'))

			print('Saved laughter segments in {}'.format(
				os.path.join(output_path, fname + '_laughter.TextGrid')))


