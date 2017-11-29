# File: generate_wavelets.py
# Author: Rosy Davis, rosydavis@ieee.org
# Last modified: 2017 Nov. 28
#
# A utility script to generate DWT image files for the image files requested.

# Parse passed-in arguments:
import argparse

# Other imports
import os

import numpy as np                     # For math and analysis
import pandas as pd                    # For data structures

import matplotlib 					   # To set backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt        # For graphing and image generation
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import sklearn as skl                  # (scikit-learn) for various machine learning tasks
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import audioread
import fma.utils as fma_utils          # Utilities provided for loading and manipulating the
									   # Free Music Archive dataset.

import pywt 						   # For wavelets
import code_timing as timer 		   # For tracking long runs




# Set control variables from args. Start by setting up:
parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help = "Directory for audio and metadata input files.")
parser.add_argument("output_dir", help = "Directory for image output files.")
parser.add_argument("-v", "--verbose", help = "Show all messages", action="store_true")
parser.add_argument("-o", "--overwrite", 
					help = "Overwrite existing files instead of skipping", 
					action="store_true")
parser.add_argument("-d", "--dwt", 
					help = "Generate DWT files", 
					action="store_true")
parser.add_argument("-c", "--cwt", 
					help = "Generate CWT files (slow)", 
					action="store_true")
parser.add_argument("-m", "--cmap", 
					help = "Specify colormap for wavelet images (defaults to 'magma')")
parser.add_argument("-x", "--wvlt_cont", 
					help = "Specify continuous wavelet (defaults to 'gaus4')")
parser.add_argument("-w", "--wvlt_disc", 
					help = "Specify discrete wavelet (defaults to 'db5')")
parser.add_argument("-z", "--size", 
					help = "Specify the dataset size to use",
					choices = ["small", "medium", "large"])
parser.add_argument("-s", "--split", 
					help = "Specify the split to use",
					choices = ["training", "validation", "test"])
parser.add_argument("-l", "--limit", help = "Limit how many files to generate", 
					type = int)
parser.add_argument("--octaves", 
					help = "Specify the number of octaves to use (defaults to 11)", 
					type = int)

# Set defaults:

# General
verbose = False       # Controls how much debugging/progress information is printed
generate_dwts = False # Should we try to generate DWT files?
generate_cwts = False # Should we try to generate CWT files (slow)?
overwrite = False     # Set to True to regenerate existing files - slow, but useful 
					  #        for code debugging; don't have to delete files between runs
cmap = "magma"        # Perceptually uniform and relatively friendly to various types of 
					  #        color-blindness, as well as greyscaling gracefully; see 
					  #        http://bids.github.io/colormap/
limit_num = None      # Set to None to run the whole set

# By default, generate training data for small dataset:
requested_subset = "small"
requested_split = "training"

# Set up the CWT (not ultimately used in this project, because the slowness of the CWT
# makes this approach less useful, but the code is provided in case it's useful):
num_octaves = 11      # 11 octaves goes from ~22 Hz to 22050 Hz, i.e. nearly the full range of 
					  # human hearing. Decreasing the number of octaves leads to loss on the 
					  # *low* end in the CWT.
wvlt_cont = 'gaus4'  

# Set up the DWT:
wvlt_disc = "db5" 

# Override as necessary from arguments:
args = parser.parse_args()
input_dir = os.path.join(args.input_dir, '')
output_dir = os.path.join(args.output_dir, '')
if args.verbose:
	verbose = args.verbose
if args.overwrite:
	overwrite = args.overwrite
if args.dwt:
	generate_dwts = args.dwt
if args.cwt:
	generate_cwts = args.cwt
if args.cmap:
	cmap = args.cmap
if args.limit:
	limit_num = args.limit
if args.wvlt_cont:
	wvlt_cont = args.wvlt_cont
if args.wvlt_disc:
	wvlt_disc = args.wvlt_disc
if args.octaves:
	num_octaves = args.octaves
if args.size:
	requested_subset = args.size
if args.split:
	requested_split = args.split



# Wrapper to print that mutes output if we're not in verbose mode:
def printt(*args, verbose = verbose, **kwargs):
	if (verbose):
		print(*args, **kwargs)
	with open("generate_wavelets.log","a+") as f:
		print(*args, **kwargs, file=f)


def flatten(arr):
	flat = np.ndarray(0,dtype = arr[0].dtype)
	for item in arr:
		flat = np.append(flat, item)
	return flat

def strip_filename(filename):
	# "/p/a/t/h/name.mp3" => "name" :
	stripped = os.path.splitext(os.path.basename(filename))[0] 
	# Now append the logical subdirectory (first 3 digit characters):
	stripped = os.path.join(stripped[0:3], stripped)
	# Note that this is missing an extension--we will add info and extension in save:
	return stripped

def fileify(stripped, subdir, tail):
	fullpath = os.path.join(output_dir, os.path.join(subdir, stripped + tail))
	
	# Make sure the directory chain exists
	directory = os.path.dirname(fullpath)
	
	if not os.path.exists(directory):
		os.makedirs(directory)
		
	return fullpath

# Adapted from https://stackoverflow.com/questions/16482166/basic-plotting-of-wavelet-
#            analysis-output-in-matplotlib
def dwtplots(track_id, stripped, tree):
	files_generated = {}
	
	# Make a new figure, since matplotlib defaults to hold on
	fig = plt.figure()
	
	# Generate the training images first
	fig.set_figwidth(small_img_dim)
	fig.set_figheight(small_img_dim)
	
	# Set the axes to not have any bordering whitespace
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	# DWT plots are by level:
	bottom = 0
	tree_arr = flatten(tree)
	vmin = np.amin(tree_arr)
	vmax = np.amax(tree_arr)
	
	ax.set_ybound([1,scales_max])
	ax.set_autoscale_on(False)
	ax.set_autoscalex_on(True) 
	
	scale = scales_max/len(tree) # Use log y scale
	for row in range(0, len(tree)):
		row_data = tree[row]
		row_data = row_data.reshape(1,len(row_data))
		cax = ax.imshow(row_data,
						cmap="magma",
						interpolation = 'none',
						vmin = vmin,
						vmax = vmax,
						aspect="auto",
						extent = [0, 30, bottom, bottom + scale],
		)

		bottom += scale

	# Save with no axis labels and no bordering whitespace 
	files_generated[(track_id, "small_dwt_noframe")] = fileify(stripped, "dwt/noframe/", "_small.png")
	files_generated[(track_id, "large_dwt_noframe")] = fileify(stripped, "dwt/noframe/", "_large.png")
	
	plt.savefig(files_generated[(track_id, "small_dwt_noframe")])
	fig.set_dpi(scales_max) # make bigger
	fig.set_figwidth(large_img_dim) 
	fig.set_figheight(scales_max/scales_max) 
	plt.savefig(files_generated[(track_id, "large_dwt_noframe")])

	# Resize and update the axes and colorbar to show, and tweak the tick labels:
	files_generated[(track_id, "small_dwt_frame")] = fileify(stripped, "dwt/frame/", "_small.png")
	files_generated[(track_id, "large_dwt_frame")] = fileify(stripped, "dwt/frame/", "_large.png")
	
	fig.set_figwidth(small_img_dim+colorbar_pad)
	cbar = fig.colorbar(cax, ticks=[vmin, vmax])
	ax.set_ylabel("DWT Level")
	ax.set_xlabel("Time [sec]")
	ax.set_yticks(np.arange(0, scales_max+1, scales_max/4))#np.append([1],(np.geomspace(1, scales_max, num_octaves))[-3:])) 
	ax.set_xticks(np.arange(0, 31, 10)) # Clips are 30s long
	ax.set_axis_on()
	fig.add_axes(ax)
	for tick in ax.get_xticklabels():
		tick.set_rotation(30)

	# Save with axis labels and minimal bordering whitespace
	plt.savefig(files_generated[(track_id, "small_dwt_frame")], bbox_inches="tight")
	fig.set_figwidth(large_img_dim+colorbar_pad)
	plt.savefig( files_generated[(track_id, "large_dwt_frame")], bbox_inches="tight")

	# Free memory by closing this figure
	plt.close(fig)
	
	return files_generated

def cwtplots(track_id, stripped, data, t, frequencies):
	files_generated = {}
	
	# Make a new figure, since matplotlib defaults to hold on
	fig = plt.figure()
	
	# Generate the training images first
	fig.set_figwidth(small_img_dim)
	fig.set_figheight(small_img_dim)
	
	# Set the axes to not have any bordering whitespace
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	# Plots are frequency versus time:
	vmax = np.max(data)
	vmin = np.min(data)

	t0 = np.min(t)
	tlast = np.max(t)
	f0 = np.min(frequencies)
	flast = np.max(frequencies)
	cax = ax.imshow(data,
					cmap=cmap,
					interpolation = 'none',
					vmin = vmin,
					vmax = vmax,
					aspect="auto", 
					extent=[t0, tlast, f0, flast],
				   )

	# Save with no axis labels and no bordering whitespace 
	files_generated[(track_id, "small_cwt_noframe")] = fileify(stripped, "cwt/noframe/", "_small.png")
	files_generated[(track_id, "large_cwt_noframe")] = fileify(stripped, "cwt/noframe/", "_large.png")
	
	plt.savefig(files_generated[(track_id, "small_cwt_noframe")])
	fig.set_dpi(scales_max) # make bigger
	fig.set_figwidth(large_img_dim) 
	fig.set_figheight(scales_max/scales_max) 
	plt.savefig(files_generated[(track_id, "large_cwt_noframe")])

	# Resize and update the axes and colorbar to show, and tweak the tick labels:
	files_generated[(track_id, "small_cwt_frame")] = fileify(stripped, "cwt/frame/", "_small.png")
	files_generated[(track_id, "large_cwt_frame")] = fileify(stripped, "cwt/frame/", "_large.png")
	
	fig.set_figwidth(small_img_dim+colorbar_pad)
	cbar = fig.colorbar(cax, ticks=[vmin, vmax])
	ax.set_ylabel("Frequency [Hz]")
	ax.set_xlabel("Time [sec]")
	ax.set_yticks([20, 7500, 15000, 22050]) # Pretty nicely spaced along audible freq. range
	ax.set_xticks(np.arange(0, 31, 10)) # Clips are 30s long
	ax.set_axis_on()
	fig.add_axes(ax)
	for tick in ax.get_xticklabels():
		tick.set_rotation(30)

	# Save with axis labels and minimal bordering whitespace
	plt.savefig(files_generated[(track_id, "small_cwt_frame")], bbox_inches="tight")
	fig.set_figwidth(large_img_dim+colorbar_pad)
	plt.savefig(files_generated[(track_id, "large_cwt_frame")], bbox_inches="tight")

	# Free memory by closing this figure
	plt.close(fig)
	
	return files_generated
	

def make_wavelets(track_id, 
				  min_f, max_f, filename,
				  total_times,
				  file_load_times,
				  cwt_times,
				  dwt_times,
				  pyplot_cwt_times,
				  pyplot_dwt_times):
	timer.tic("single_run")
	files_generated = {}
	
	
	
	# Make sure the audio file exists
	# print("File:", filename)
	assert(os.path.isfile(filename))
	
	# Get the part of the filename that gets replicated in image file names
	stripped = strip_filename(filename)
	
	# If overwrite is not enabled, we decide whether to generate files based on:
	#   a) whether or not a particular wavelet (DWT/CWT) is requested, and
	#   b) whether or not all files for that wavelet (or those wavelets) already exists
	if not overwrite:
		cwt_to_generate = False
		if (not os.path.exists(fileify(stripped, "cwt/noframe/", "_small.png")) or  
				not os.path.exists(fileify(stripped, "cwt/noframe/", "_large.png")) or  
				not os.path.exists(fileify(stripped, "cwt/frame/", "_small.png")) or  
				not os.path.exists(fileify(stripped, "cwt/frame/", "_large.png"))): # one or more CWT files is missing
			if generate_cwts:
				cwt_to_generate = True

		dwt_to_generate = False
		if (not os.path.exists(fileify(stripped, "dwt/noframe/", "_small.png")) or  
				not os.path.exists(fileify(stripped, "dwt/noframe/", "_large.png")) or  
				not os.path.exists(fileify(stripped, "dwt/frame/", "_small.png")) or  
				not os.path.exists(fileify(stripped, "dwt/frame/", "_large.png"))): # one or more DWT files is missing
			if generate_dwts:
				dwt_to_generate = True
	# If overwrite is enabled, always generate the requested type(s) of wavelet images:
	else:
		cwt_to_generate = generate_cwts # for this file <- for all files
		dwt_to_generate = generate_dwts # for this file <- for all files

	files_to_generate = cwt_to_generate or dwt_to_generate # for this file

	#print("There {} files to generate.".format("are a nonzero number of" if files_to_generate else "are no"))
	
	if (files_to_generate):
		# Load and adjust data
		timer.tic("file_load")
		try:
			data, sample_rate = librosa.load(filename, sr=None, mono=True) # Convert to mono for 
																		   # simpler processing
			# Set up time/sample time variables
			dt = 1/sample_rate
			t = dt*np.arange(len(data)) # Time of data samples in seconds [x axis]

			# Normalize the (signed) data relative to its max value:
			data = 1/(np.max(abs(data))+np.finfo(float).eps)*data
		except audioread.NoBackendError as e:
			printt("Couldn't load {} because the backend is missing.".format(filename))
			printt(e)
			raise
		except Exception as e: # all other errors, just log, then skip this file
			printt("\n\n=> Couldn't load {}:\n{}\n".format(filename,e), verbose = False)
			cwt_to_generate = False # force code to not do anything else for this file
			dwt_to_generate = False

		file_load_times = np.append(file_load_times, timer.toc("file_load"))

		
		
		# Generate CWT graphics
		if cwt_to_generate:
			# Calculate and adjust CWT:
			timer.tic("cwt")
			[cwt_data,frequencies] = pywt.cwt(data, scales, wvlt_cont, dt)
			# Convert CWT data from straight magnitude to power (dB)
			cwt_data = 10*np.log10(np.abs(cwt_data)**2 + np.finfo(float).eps)
			cwt_times = np.append(cwt_times, timer.toc("cwt"))
			
			# Find the minimum and maximum frequencies, so we can make sure we're operating on
			# the same frequency scale for all generated images (i.e. comparing apples to apples):
			min_f = min(frequencies[0],frequencies[-1])
			max_f = max(frequencies[0],frequencies[-1])

			# Plot and save CWT images:
			timer.tic("pyplot_cwt")
			files_generated = {**files_generated,
							   **cwtplots(track_id, stripped, cwt_data, t, frequencies)}
			pyplot_cwt_times = np.append(pyplot_cwt_times, timer.toc("pyplot_cwt"))
		
		
		
		# Generate DWT graphics
		if dwt_to_generate:
			# Calculate and adjust DWT:
			timer.tic("dwt")
			tree = pywt.wavedec(data, wvlt_disc)
			# Convert DWT data from straight magnitude to power (dB)
			for row in range(0, len(tree)):
				tree[row] = 10*np.log10(np.abs(tree[row])**2 + np.finfo(float).eps)
			dwt_times = np.append(dwt_times, timer.toc("dwt"))
			
			# Plot and save DWT images:
			timer.tic("pyplot_dwt")
			files_generated = {**files_generated,
							   **dwtplots(track_id, stripped, tree)}
			pyplot_dwt_times = np.append(pyplot_dwt_times, timer.toc("pyplot_dwt"))

	total_times = np.append(total_times, timer.toc("single_run"))

	return (min_f, max_f, files_generated, 
			total_times,
			file_load_times,
			cwt_times,
			dwt_times,
			pyplot_cwt_times,
			pyplot_dwt_times)






if not generate_cwts and not generate_dwts:
	print ("\nNo wavelet images requested...nothing to do!\n")
else:
	# Finish setting up the CWT:
	scales_max = 2**(num_octaves-1)
	scales = np.geomspace(1, scales_max, num=num_octaves).astype(int)

	# Make some tweaks for the wavelet image generation:
	figdpi = 256 
	training_dim = 256
	border_pad = 0.525 # inches--determined experimentally. The code is written so that the
					   # border (whitespace, axes, etc) on printed images extends beyond the 
					   # image as used as data, so that rescaling won't introduce artifacts. 
					   # This extra padding is used to make sure that when the border for printed
					   # images is added, the image does not need to be rescaled to fit the page.
	plt.rcParams["figure.dpi"] = 600
	plt.rcParams["figure.titlesize"] = 18
	plt.rcParams["image.interpolation"] = "nearest"
	plt.rcParams["image.cmap"] = cmap
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams["savefig.dpi"] = "figure"
	plt.rcParams["xtick.minor.visible"] = True
	plt.rcParams['xtick.labelsize'] = 6.5
	plt.rcParams['ytick.labelsize'] = 6.5
	colorbar_pad = 0.25 # inches--determined experimentally.
	plt.rcParams["figure.dpi"] = figdpi 
	small_img_dim = training_dim/figdpi # inches
	large_img_dim = 8.5-1.5-1-border_pad # inches: 8.5" paper, 1.5" margin on left and 1" on right
	# Set default to be the 1024x1024 pixel size
	plt.rcParams["figure.figsize"] = (small_img_dim, 
									  small_img_dim) 


	printt("\nGenerate run begun at {}.\n".format(timer.datetimestamp()))

	# Load the metadata files
	tracks = fma_utils.load(input_dir + 'tracks.csv')
	features = fma_utils.load(input_dir + 'features.csv')

	printt("Tracks and features loaded.")

	# Make sure everything in features is in tracks and vice versa
	np.testing.assert_array_equal(features.index, tracks.index)

	# Use the specified data subset:
	subset = tracks['set', 'subset'] <= requested_subset
	printt("{} tracks in the {} set.".format(len(tracks.loc[subset].index), 
											 requested_subset))

	# Get the pre-split sets from the FMA data set
	split = tracks['set', 'split'] == requested_split
	printt("{} tracks in the {} set.".format(len(tracks.loc[subset & split].index), 
											 requested_split))

	printt("Splits determined.")

	# Overall file generation timer
	timer.tic("total_file_gen")

	# Set up to keep track of the frequency range we're examining:
	max_f, min_f = (None, None) # dummy values, will be overwritten right away

	# Set up to track how long things take:
	total_times = np.ndarray(0)
	file_load_times = np.ndarray(0)
	cwt_times = np.ndarray(0)
	dwt_times = np.ndarray(0)
	pyplot_cwt_times = np.ndarray(0)
	pyplot_dwt_times = np.ndarray(0)

	printt()

	stopwatch = timer.tic()
	files_generated = {}
	examined = 0
	# Handle each ID in the requested set/split
	rel_track_ids = tracks.loc[subset & split].index
	num_clips = len(rel_track_ids) # done + this loop
	if limit_num is not None and num_clips > limit_num:
		num_clips = limit_num

	per = np.round(num_clips/100*0.5);
	per = max(per,1)
	for track_id in rel_track_ids:
		if (examined % per == 0):
			if (examined < num_clips): # not complete
				stopwatch = timer.toc()
				printt(f"{examined} ({examined/num_clips:0.1%}) in {timer.time_from_sec(stopwatch)}", 
					  end='')
				if (examined > 0):
					printt(f", T/W", end='')
					if generate_cwts:
						printt(f"/CWT/PCWT", end='')
					if generate_dwts:
						printt(f"/DWT/PDWT", end="")
					printt(" = ", end='')
					printt(f"{0 if len(total_times) == 0 else np.mean(total_times):0.2f}", end='')
					printt(f"/{0 if len(file_load_times) == 0 else np.mean(file_load_times):0.2f}", end='')
					if generate_cwts:
						printt(f"/{0 if len(cwt_times) == 0 else np.mean(cwt_times):0.2f}", end='')
						printt(f"/{0 if len(pyplot_cwt_times) == 0 else np.mean(pyplot_cwt_times):0.2f}", end='')
					if generate_dwts:
						printt(f"/{0 if len(dwt_times) == 0 else np.mean(dwt_times):0.2f}", end='')
						printt(f"/{0 if len(pyplot_dwt_times) == 0 else np.mean(pyplot_dwt_times):0.2f}", end='')
				printt("...", end="", flush=True)

		examined += 1
		# Extract the MP3 file name:
		filename = fma_utils.get_audio_path(input_dir, track_id)
		
		# Run the file generation:
		(mif, maf, fg, total_times, file_load_times, cwt_times,
		 dwt_times, pyplot_cwt_times, pyplot_dwt_times) = make_wavelets(track_id, min_f, max_f, 
																		filename, 
																		total_times,
																		file_load_times,
																		cwt_times,
																		dwt_times,
																		pyplot_cwt_times,
																		pyplot_dwt_times)
		
		# Adjust max/min frequency and warn if necessary--we want all images to handle the same
		# frequency range:
		if min_f is None or mif < min_f:
			if min_f is not None:
				printt(f"WARNING: adjusting min. frequency from {min_f} to "
					  f"{mif} after one or more images have already been generated.")
			min_f = mif
		if max_f is None or maf > max_f:
			if max_f is not None:
				printt(f"WARNING: adjusting max. frequency from {max_f} to "
					  f"{maf} after one or more images have already been generated.")
			max_f = maf
			
		# Record what files we generated for posterity:
		files_generated = {**fg, **files_generated}
		if limit_num is not None and examined >= limit_num:
			break
		
		
	# How did we do?
	printt(f"\n\n\nGenerated {len(files_generated)} file(s) for {examined} track(s) in "
		  f"{timer.time_from_sec(timer.toc('total_file_gen'))}."
		  f"\n-----------------------------------------------------")
	for times, name in [(total_times, "total"),
						(file_load_times, "load"), 
						(cwt_times, "CWT"),
						(dwt_times, "DWT"),
						(pyplot_cwt_times, "PyPlot for CWT"),
						(pyplot_dwt_times, "PyPlot for DWT")]:
		if (len(times) > 0):
			printt(f"* Avg. {name} time: {timer.time_from_sec(np.mean(times))}.")
			
	np.savez("fma_generation", 
			 files_generated=files_generated,
			 total_times=total_times,
			 file_load_times=file_load_times,
			 cwt_times=cwt_times,
			 dwt_times=dwt_times,
			 pyplot_cwt_times=pyplot_cwt_times,
			 pyplot_dwt_times=pyplot_dwt_times)
