import sys
import laugh_segmenter

def seconds_to_frames(s):
	return(int(s*100))

def parse_inputs():
	process = True

	try:
		a_file = sys.argv[1]
	except:
		print "Enter the audio file path as the first argument"
		process = False

	try:
		model_path = sys.argv[2]
	except:
		print "Enter the stored model path as the second argument"
		process = False

	try:
		output_audio_path = sys.argv[3]
	except:
		print "Enter the output audio path as the third argument"
		process = False

	try:
		threshold = float(sys.argv[4])
	except:
		threshold= 0.5

	try:
		min_length = float(sys.argv[5])
	except:
		min_length = 0.2

	if process:
		return (a_file, model_path, output_audio_path, threshold, min_length)
	else:
		return False

# Usage: python segment_laughter.py <input_audio_file> <stored_model_path> <output_folder>

if __name__ == '__main__':
	if parse_inputs():
		input_path, model_path, output_path, threshold, min_length = parse_inputs()
		min_length = seconds_to_frames(min_length)

		laughs = laugh_segmenter.segment_laughs(input_path,model_path,output_path,threshold,min_length) 
		print; print "found %d laughs." % (len (laughs))
		for laugh in laughs:
			print laugh
