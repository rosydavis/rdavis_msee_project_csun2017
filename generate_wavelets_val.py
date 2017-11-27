# Control variables
verbose = True        # Controls how much debugging/progress information is printed
overwrite = False     # Set to True to regenerate existing files - slow, but useful 
					  #        for code debugging; don't have to delete files between runs
seed = 42             # Used to seed the randomizer predictably; set to None to auto-generate
cmap = "magma"        # Perceptually uniform and relatively friendly to various types of 
					  #        color-blindness, as well as greyscaling gracefully; see 
					  #        http://bids.github.io/colormap/
generate_dwts = True  # Try to generate DWT files
generate_cwts = False # Try to generate CWT files (slow)
limit_num = None      # Set to None to run the whole set

# Set up the CWT:
num_octaves = 11      # 11 octaves goes from ~22 Hz to 22050 Hz, i.e. nearly the full range of 
					  # human hearing. Decreasing the number of octaves leads to loss on the 
					  # *low* end in the CWT.
wvlt_cont = 'gaus4'  

# Set up the DWT:
wvlt_disc = "db5" 

# This block is adapted from FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.
# provided usage script
import os

#import IPython.display as ipd          # For ???
import numpy as np                     # For math and analysis
import pandas as pd                    # For data structures
import matplotlib.pyplot as plt        # For graphing
# import seaborn as sns
import sklearn as skl                  # (scikit-learn) for various machine learning tasks
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import audioread
import fma.utils as fma_utils          # Utilities provided for loading and manipulating the
									   # Free Music Archive dataset.

#########

# # Do necessary imports:
# import random
# import scipy.signal
# matplotlib.use('Agg') # don't show images to user--we're just going to save them

# import os.path

# # Data set
# import fma.utils as utils
# import IPython.display as ipd

# import scipy.io.wavfile as wavf
# import wave
import pywt
import net_visualization as viz
import code_timing as timer

# Finish setting up the CWT:
scales_max = 2**(num_octaves-1)
scales = np.geomspace(1, scales_max, num=num_octaves).astype(int)

# Set up the vizualization module to generate images that will match the style of other
# graphics in the report:
viz.setup()
# ...and make some tweaks specifically for the wavelet image generation:
figdpi = 256 
training_dim = 256
border_pad = 0.525 # inches--determined experimentally. The code is written so that the
				   # border (whitespace, axes, etc) on printed images extends beyond the 
				   # image as used as data, so that rescaling won't introduce artifacts. 
				   # This extra padding is used to make sure that when the border for printed
				   # images is added, the image does not need to be rescaled to fit the page.
plt.rcParams['xtick.labelsize'] = 6.5
plt.rcParams['ytick.labelsize'] = 6.5
colorbar_pad = 0.25 # inches--determined experimentally.
plt.rcParams["figure.dpi"] = figdpi 
small_img_dim = training_dim/figdpi # inches
large_img_dim = 8.5-1.5-1-border_pad # inches: 8.5" paper, 1.5" margin on left and 1" on right
# Set default to be the 1024x1024 pixel size
plt.rcParams["figure.figsize"] = (small_img_dim, 
								  small_img_dim) 

# Wrapper to print that mutes output if we're not in verbose mode:
def printt(*args, verbose = verbose, **kwargs):
	if (verbose):
		print(*args, **kwargs)
	with open("generate_wavelets.log","a+") as f:
		print(*args, **kwargs, file=f)

# Adapted from fma usage code:
AUDIO_DIR = "/Volumes/MEDIA/fma_large/"
IMAGE_DIR = "/Volumes/MEDIA/fma_large/data/"

printt("\nGenerate run begun at {}.\n".format(timer.datetimestamp()))

# Load the metadata files
tracks = fma_utils.load(AUDIO_DIR + 'tracks.csv')
features = fma_utils.load(AUDIO_DIR + 'features.csv')

printt("Tracks and features loaded.")

# Make sure everything in features is in tracks and vice versa
np.testing.assert_array_equal(features.index, tracks.index)

# echonest and genres are not currently used:
# echonest = fma_utils.load(AUDIO_DIR + 'echonest.csv')
# genres = fma_utils.load(AUDIO_DIR + 'genres.csv')

# # Make sure everything in echonest is in tracks
# assert echonest.index.isin(tracks.index).all()

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
	fullpath = os.path.join(IMAGE_DIR, os.path.join(subdir, stripped + tail))
	
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
	ax.set_ylabel(viz.axislabel("DWT Level"))
	ax.set_xlabel(viz.axislabel("Time [sec]"))
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

	# Plots are time versus frequency:
	vmax = np.max(data)
	vmin = np.min(data)
	# OLD APPROACH
#     cax = ax.contourf(t, np.log2(frequencies), data, cmap="jet", extend = "both", 
#                       vmax = vmax, vmin = vmin)
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
	ax.set_ylabel(viz.axislabel("Frequency [Hz]"))
	ax.set_xlabel(viz.axislabel("Time [sec]"))
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
	assert(os.path.isfile(filename))
	
	# Get the part of the filename that gets replicated in image file names
	stripped = strip_filename(filename)
	
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

	else:
		cwt_to_generate = generate_cwts # for this file <= for all files
		dwt_to_generate = generate_dwts # for this file <= for all files

	files_to_generate = cwt_to_generate or dwt_to_generate # for this file

	# print("There {} files to generate.".format("are a nonzero number of" if files_to_generate else "are no"))
	
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

# print(tracks.columns.values)
# print(tracks['set','subset'].unique())

# Code adapted from FMA (commenting mine):
printt("Functions defined.")

# # Use the small data set (which is balanced re: genre and <8GB):
# Use the large data set (which is not balanced re: genre but has much more data):
large = tracks['set', 'subset'] <= 'large'
printt("{} tracks in the large set.".format(len(large)))

# Get the pre-split sets from the FMA data set
train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

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
# Handle each ID in the requested set
train_track_ids = test.loc[large & test].index
num_clips = len(train_track_ids) # done + this loop
if limit_num is not None and num_clips > limit_num:
	num_clips = limit_num

per = np.round(num_clips/100*0.5);
per = max(per,1)
for track_id in train_track_ids:
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
	filename = fma_utils.get_audio_path(AUDIO_DIR, track_id)
	
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
