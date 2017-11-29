# File: move_images.py
# Author: Rosy Davis, rosydavis@ieee.org
# Last modified: 2017 Nov. 28
#
# A utility script to copy DWT images from a folder that keeps them placed by file name
# (as is true of the source MP3s in the FMA dataset) to folders that split them by dataset
# split (test, train, val) and genre (folk, hip-hop, et cetera).  
#
# Note that this does not move the source files, but instead copies them. Wavelet image
# files are small, and this ensures that the source images remain in place so they can be
# reused. For example, for the FMA dataset, which has three differently-sized subsets, any
# training image in the "small" dataset will also appear as a training image in the
# "large" dataset. By copying instead of moving, the source image will remain at the path
# equivalent to the path for the source audio, and can be reused if it is desirable to 
# work with both the small and the large datasets.

# Parse passed-in arguments:
import argparse

# File system utilities:
import os
import shutil

# Used for error checking:
import numpy as np

# FMA dataset utilities
import fma.utils as fma_utils          # Utilities provided for loading and manipulating the
									   # Free Music Archive dataset.

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", 
					help = "Directory of images currently stored at FMA-style paths.")
parser.add_argument("output_dir", 
					help = "Directory of images to be saved in a by-class hierarchy.")
parser.add_argument("-z", "--size", 
					help = "Specify the dataset size to use",
					choices = ["small", "medium", "large"])
parser.add_argument("-s", "--split", 
					help = "Specify the split to use",
					choices = ["training", "validation", "test"])
parser.add_argument("-w", "--wavelet", 
					help = "Specify the wavelet type to use",
					choices = ["dwt", "cwt"])

# By default, generate training data for small dataset:
requested_subset = "small"
requested_split = "training"
requested_wavelet = "dwt"

# Override as necessary from arguments:
args = parser.parse_args()
input_dir = os.path.join(args.input_dir, '')
output_dir = os.path.join(args.output_dir, '')
if args.size:
	requested_subset = args.size
if args.split:
	requested_split = args.split
if args.wavelet:
	requested_wavelet = args.wavelet

if requested_split == "training":
	requested_split_path = "train"
elif requested_split == "validation":
	requested_split_path = "validation"
elif requested_split == "test":
	requested_split_path = "test"





# Load the metadata files
tracks = fma_utils.load(input_dir + 'tracks.csv')
features = fma_utils.load(input_dir + 'features.csv')

# Make sure everything in features is in tracks and vice versa
np.testing.assert_array_equal(features.index, tracks.index)

# Use the specified data subset:
subset = tracks['set', 'subset'] <= requested_subset
split = tracks['set', 'split'] == requested_split
rel_track_ids = tracks.loc[subset & split].index

y_values = tracks.loc[subset & split, ('track', 'genre_top')]
unique_genres = y_values.unique().categories






# Copy files:
for track_id in rel_track_ids:
	try:
		y_str = y_values.loc[track_id].lower()
	except:
		# print("Skipping {}; bad genre...".format(track_id))
		continue
	
	trackstr = "{:06d}".format(track_id)

	try:
		curr_path = os.path.join(input_dir, 
						os.path.join(requested_wavelet,
							os.path.join("noframe",
								os.path.join(trackstr[0:3],
											 "{}_small.png".format(trackstr)))))
		assert(os.path.isfile(curr_path))
	except:
		# print("Skipping {}; file '{}' not found...".format(track_id, curr_path))
		continue
	# print(curr_path) 
	
	new_path = os.path.join(output_dir, 
					os.path.join("byclass",
						 os.path.join(requested_subset,
							 os.path.join(requested_wavelet,
								 os.path.join(requested_split_path, 
									  os.path.join(y_str, "{}.png".format(trackstr)))))))
	# print(new_path) 
	
	directory = os.path.dirname(new_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

	shutil.copyfile(curr_path, new_path)
		