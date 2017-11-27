# Keras imports
# TODO: docs -  https://keras.io/preprocessing/image/
import keras
import keras.preprocessing.image
import keras.applications as keras_apps
from keras import backend as K # for clearing tensorflow session
K.clear_session() # Just in case we've started out with garbage in there

# Tensorflow (logging/tracing):
import tensorflow as tf
from tensorflow.python.client import timeline # http://bit.ly/2ybeTIk, 
											  # http://bit.ly/2gpXphX

# JSON (tensorflow logging/tracing):
import json

# Other imports - 
# Audio alerts/path manipulation:
import os

# Set timezone:
import time
os.environ['TZ'] = 'US/Pacific'
time.tzset()

# General:
import math
import random
import numpy as np
import scipy as sp
import pandas as pd
import code_timing as timer 

# My utilities:
import utilities as ut 

RESULTS_COLS = ["Run Started",
			    "Source Processor",
				"Source",
				"Pass Epochs",
				"Batch Size",
				"Steps Per Epoch",
				"Validation Steps Per Epoch",
				"Data Augmentation Factor",
				"Data Set Size",
				"Wavelet",
				"Learning Rate",
				"Learning Rate Decay",
				"Rho",
				"Epsilon",
				"Model",
				"Run Duration",
				"Final Training Accuracy",
				"Final Validation Accuracy",
				"Training Loss History",
				"Validation Loss History",
				"Training Accuracy History",
				"Validation Accuracy History",
				]




def formatted(tuple_key):
	index = 0
	s = ""
	for elem in tuple_key:
		if (index > 0):
			s += "x"
		s += "{}".format(elem)
		index += 1
	return s

def run_key(param_dict, run_started):
	return (param_dict["spu"],
			param_dict["source"],
			param_dict["pass_epochs"],
			param_dict["batch_size"],
			param_dict["steps_per_epoch"],
			param_dict["validation_steps"],
			param_dict["augmentation"],
			param_dict["which_size"],
			param_dict["which_wavelet"])

def assign_run_key(series_obj, param_dict, run_started):
	series_obj["Run Started"] = run_started
	series_obj["Source Processor"] = param_dict["spu"]
	series_obj["Source"] = param_dict["source"]
	series_obj["Pass Epochs"] = param_dict["pass_epochs"]
	series_obj["Batch Size"] = param_dict["batch_size"]
	series_obj["Steps Per Epoch"] = param_dict["steps_per_epoch"]
	series_obj["Validation Steps Per Epoch"] = param_dict["validation_steps"]
	series_obj["Data Augmentation Factor"] = param_dict["augmentation"]
	series_obj["Data Set Size"] = param_dict["which_size"]
	series_obj["Wavelet"] = param_dict["which_wavelet"]

def recover_run_key(series_obj):
	return (series_obj["Source Processor"],
			(int)(series_obj["Source"]),
			(int)(series_obj["Pass Epochs"]),
			(int)(series_obj["Batch Size"]),
			(int)(series_obj["Steps Per Epoch"]),
			(int)(series_obj["Validation Steps Per Epoch"]),
			("0" if series_obj["Data Augmentation Factor"] == 0 else 
				("0.5" if series_obj["Data Augmentation Factor"] == 0.5 else
				"{:0.2f}".format(series_obj["Data Augmentation Factor"]))),
			series_obj["Data Set Size"],
			series_obj["Wavelet"])
		
def assign_opt_key(series_obj, opt_key):
	series_obj["Learning Rate"] = opt_key[0]
	series_obj["Learning Rate Decay"] = opt_key[1]
	series_obj["Rho"] = opt_key[2]
	series_obj["Epsilon"] = opt_key[3]
		
def assign_results_history(series_obj, model_key, runsec, hist):
	series_obj["Model"] = model_key
	series_obj["Run Duration"] = runsec
	series_obj["Final Training Accuracy"] = hist.history["categorical_accuracy"][-1]
	series_obj["Final Validation Accuracy"] = hist.history["val_categorical_accuracy"][-1]
	series_obj["Training Loss History"] = hist.history["loss"]
	series_obj["Validation Loss History"] = hist.history["val_loss"]
	series_obj["Training Accuracy History"] = hist.history["categorical_accuracy"]
	series_obj["Validation Accuracy History"] = hist.history["val_categorical_accuracy"]
	
	return series_obj["Final Training Accuracy"], series_obj["Final Validation Accuracy"]





def save_timeline(param_dict):
	trace = timeline.Timeline(step_stats=param_dict["run_metadata"].step_stats)
	ut.ensure_dir(param_dict["run_timelines_file"])
	json_str = trace.generate_chrome_trace_format()
	json_obj_new = json.loads(json_str)
	
	try:
		with open(param_dict["run_timelines_file"], 'r') as f:
			json_obj_old = json.load(f)
	except Exception as e:
		# # Usually exceptions are just because we need to create the file and can be 
		# # safely skipped/ignored:
		# print("ALERT: Could not load JSON: ",e)
		json_obj_old = {}
		json_obj_old["traceEvents"] = []
		
	json_obj_new["traceEvents"] = (json_obj_old["traceEvents"] + 
								   json_obj_new["traceEvents"])
	json_str = json.dumps(json_obj_new, indent=4)
	with open(param_dict["run_timelines_file"], 'w') as f:
		f.write(json_str)





def calc_img_stats(directory):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(directory):
		for fn in filenames:
			fn = os.path.join(dirpath,fn)
			f.append(fn)
			
	# Mean:
	count = 0
	for pth in f:
		try:
			asdata = sp.ndimage.imread(pth).astype("float64")
			if count == 0:
					mean_img = asdata
			else:
					mean_img += asdata
			count += 1

			del asdata # trigger some garbage collection
		except:
			pass
	mean_img *= 1/count 
			  
	# Std. Dev.:
	count = 0
	for pth in f:
		try:
			asdata = sp.ndimage.imread(pth).astype("float64")
			if count == 0:
					std_dev = (asdata - mean_img)**2
			else:
					std_dev += (asdata - mean_img)**2
			count += 1

			del asdata # trigger some garbage collection
		except:
			pass
	std_dev = ((std_dev)**(1/2))/count
	
	return mean_img, std_dev

# https://github.com/fchollet/keras/issues/3679
def fit_from_directory(obj, mean=None, std=None):
	''' Required for featurewise_center, featurewise_std_normalization 
		when using image from directory.

	# Arguments
	mean: Mean Pixels of training database
	std: Standard deviation of training database
	'''
	if obj.featurewise_center:
		obj.mean = mean

	if obj.featurewise_std_normalization:
		obj.std = std

def set_up_generators(param_dict):
	print("Creating generators with batch size {}...".format(param_dict["batch_size"]))

	# Calculate the mean and std. dev. for the training set (slow, but we only have to 
	# do it once):
	fma_stats_file = "fma_{}_{}_stats.npz".format(param_dict["which_size"], 
														  param_dict["which_wavelet"])
	fma_stats_file = os.path.join("saved_objects", fma_stats_file)
	if os.path.exists(fma_stats_file):
		print(("Loading mean and standard deviation for the training set from "
			   "file '{}'.\n").format(fma_stats_file))
		with np.load(fma_stats_file) as data:
			param_dict["mean_img"] = data["mean_img"]
			param_dict["std_dev"] = data["std_dev"]
	else:
		print(("Calculating mean and standard deviation for the {} {} "
			   "training set...").format(param_dict["which_size"], 
								 		 param_dict["which_wavelet"].upper()), end="")
		(param_dict["mean_img"], 
		 param_dict["std_dev"]) = calc_img_stats(os.path.join(
												 param_dict["img_dir"], 
												 "train"))
		# Drop the alpha channel
		param_dict["mean_img"] = param_dict["mean_img"][:, :, :3]
		param_dict["std_dev"] = param_dict["std_dev"][:, :, :3]
		np.savez(fma_stats_file, mean_img = param_dict["mean_img"], 
				 std_dev = param_dict["std_dev"])
		print("done!\n")
			
	# Then flow data from file system, using the mean and std. dev. calculated.

	# Unless param_dict["augmentation"] is > 0, we're not actually going to do any 
	# real data augmentation, but we always want to subtract the mean and normalize:
	data_gen_args = dict(featurewise_center=True,
											 featurewise_std_normalization=True)
	
	# If we are doing data augmentation, restrict the augmentation to transforms that
	#  make sense for our *WT-of-audio files, i.e. time shift only:
	## Only apply data augmentation (if it's being used) to training data:
	data_gen_args_aug = data_gen_args.copy()
	if param_dict["augmentation"] > 0:
		print("Using up to {:0.1%} horizontal shift to augment training data.".format(
															param_dict["augmentation"]))
		data_gen_args_aug["width_shift_range"]=param_dict["augmentation"] 
										# horizontal (time) shift--not vertical 
										# shift, which would be equivalent to 
										# shifting the spectrum.
		data_gen_args_aug["fill_mode"]="wrap"
	train_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args_aug)
	fit_from_directory(train_datagen, mean = param_dict["mean_img"], 
					   std = param_dict["std_dev"])

	## No augmentation for validation/testing data:
	val_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
	test_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

	# The test generator should generate the whole dataset
	test_gen_size = (800 if param_dict["which_size"]=="small" else 4651)

	# The fit_from_directory method gets around the fact that .fit only works for  
	# numpy data, and even the small FMA data set is too large for local memory--it
	# sets the mean/std. dev. images manually:
	fit_from_directory(val_datagen, mean = param_dict["mean_img"], 
					   std = param_dict["std_dev"])
	fit_from_directory(test_datagen, mean = param_dict["mean_img"], 
					   std = param_dict["std_dev"])

	# Use the generators to flow data from the relevant directory
	train_image_generator = train_datagen.flow_from_directory(
										os.path.join(param_dict["img_dir"], 'train/'),
										class_mode="categorical",
										seed=param_dict["seed"],
										batch_size=param_dict["batch_size"])
	val_image_generator = val_datagen.flow_from_directory(
									os.path.join(param_dict["img_dir"], 'validation/'),
									class_mode="categorical",
									seed=param_dict["seed"],
									batch_size=param_dict["batch_size"])

	test_image_generator = test_datagen.flow_from_directory(
										os.path.join(param_dict["img_dir"], 'test/'),
										class_mode="categorical",
										seed=param_dict["seed"],
										# These are both necessary to make the test 
										# generator see every image exactly once (it
										# only can generate images of the same class in a
										# single batch, so for the extended dataset, where
										# the classes aren't balanced and have odd numbers
										# of elements, it'll wrap on some classes and skip 
										# elements on others if batch_size is > 1)
										shuffle=False,
										batch_size=1) 
	return (train_image_generator, val_image_generator, test_image_generator)













def stack_two_layer_block(param_dict, x):
	# Affine layer:
	x = keras.layers.Dense(param_dict["hidden_size"],
						   kernel_initializer='random_uniform',
						   bias_initializer='zeros')(x) 
	# Batch normalization:
	x = keras.layers.normalization.BatchNormalization()(x)
	# ReLU activation:
	x = keras.layers.Activation('relu')(x) 
	# Dropout:
	x = keras.layers.Dropout(0.5)(x)
	# Hidden layer / Softmax Classifier
	predict = keras.layers.Dense(param_dict["num_classes"],       
								 activation='softmax',
								 kernel_initializer='random_uniform',
								 bias_initializer='zeros',
								 kernel_regularizer=keras.regularizers.l2(0.0001))(x)
		
	return predict









def run_pretrained_model(param_dict, generators, models, opts, 
						 model_class, model_key, opt_key, 
						 print_layers, freeze_to, eta, save_weights = True,
						 epoch_batch_size = 5):
	assert(epoch_batch_size >= 1)
	try:
		len(freeze_to)
	except:
		freeze_to = [freeze_to] # turn single elements into a one-item list

	print("Using optimizer {}...".format(opt_key))
	
	timer.tic()
	run_started = timer.datetimestamp()
	print(("{} run begun at {}."
		   "\n\t[{} epochs (x{} passes) on {} FMA on {} takes"
		   "\n\t{}.]\n").format(param_dict["model_names"][model_key], 
		   						   run_started, 
		   						   param_dict["pass_epochs"], 
		   						   len(freeze_to) + 1,
		   						   param_dict["which_size"],
		   						   param_dict["spu"].upper(),
		   						   eta))

	# TODO: a lot of this code and commenting is from Keras docs; credit appropriately
	# https://keras.io/applications/

	# Get the pre-trained base model, without the top layer (because our input is a 
	# different shape), using the trained weights for ImageNet, to use as a starting 
	# point:
	basemodel = model_class(include_top=False, 
							input_shape=param_dict["mean_img"].shape, 
							weights='imagenet')
	x = basemodel.output
	# Add a global spatial average pooling layer at the output: [TODO: explain what  
	# this does]
	x = keras.layers.GlobalAveragePooling2D()(x)
	
	# Add Affine/BatchNorm/ReLU/Dropout/Affine-softmax-categorization block:
	predict = stack_two_layer_block(param_dict, x)

	# Now make the model:
	models[model_key] = keras.models.Model(basemodel.input, predict)

	# Train only the top layers (which were randomly initialized) while freezing
	# all convolutional layers (which were pretrained on ImageNet):
	for layer in basemodel.layers:
		layer.trainable = False

	# Compile the model (must be done after setting layer trainability):
	models[model_key].compile(optimizer = opts[opt_key], 
													  **param_dict["compile_args"])

	# Train just the classifier for the requested number of epochs:
	print("First-round training (training the classifier)...")
	initial_epoch = 0
	results = None
	# This loop of multiple checkpoints helps with memory management, esp. for VGG16/
	#     VGG19, which have a huge number of parameters - see also
	#     http://bit.ly/2hDHJay for more information.
	while initial_epoch < param_dict["pass_epochs"]:
		# Split into "epoch_batch_size"-epoch training batches
		final_epoch = min(initial_epoch+epoch_batch_size,
						  param_dict["pass_epochs"])
		print("\nTraining for epochs {} to {}...".format(initial_epoch+1, final_epoch))
		results_new = models[model_key].fit_generator(
									generators["train"],
									validation_data=generators["val"],
									verbose=param_dict["run_verbosity"], 
									epochs=final_epoch,
									steps_per_epoch=param_dict["steps_per_epoch"], 
									validation_steps=param_dict["validation_steps"],
									use_multiprocessing=True,
									initial_epoch=initial_epoch)

		# Merge these new results with existing results for previous batches:
		if results is not None:
			# Merge the two results lists:
			for key in results.history:
				results.history[key].extend(results_new.history[key])
		else:
			results = results_new

		# Now start from where we stopped on this round
		initial_epoch = final_epoch

	# At this point, the top layers are well trained and we can start fine-tuning
	# convolutional layers from Xception. We will freeze the bottom N layers
	# and train the remaining top layers.

	# Visualize layer names and layer indices to see how many layers we should freeze:
	if print_layers:
		for i, layer in enumerate(models[model_key].layers):
			print(i, layer.name)

	pass_num = 1
	for freeze in freeze_to:
		pass_num += 1
		# Freeze all layers up to the specified value; unfreeze everything 
		#     after (and including): 
		for layer in models[model_key].layers[:freeze]:
			layer.trainable = False
		for layer in models[model_key].layers[freeze:]:
			layer.trainable = True

		# we need to recompile the model for these modifications to take effect
		# we use SGD with a low learning rate because SGD trains more slowly than RMSprop  
		# (a good thing, in this case): [TODO: check this]
		models[model_key].compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), 
														  **param_dict["compile_args"])

		# Train again for the requested number of epochs:
		print(("\n\nFurther training (refining convolutional blocks, starting with"
			   "\n\tlayer {})...").format(freeze))
		while initial_epoch < pass_num*param_dict["pass_epochs"]:
			# Split into "epoch_batch_size"-epoch training batches
			final_epoch = min(initial_epoch+epoch_batch_size,
							  pass_num*param_dict["pass_epochs"])
			print("\nTraining for epochs {} to {}...".format(initial_epoch+1, final_epoch))
			results_new = models[model_key].fit_generator(
										generators["train"],
										validation_data=generators["val"],
										verbose=param_dict["run_verbosity"], 
										epochs=final_epoch,
										steps_per_epoch=param_dict["steps_per_epoch"], 
										validation_steps=param_dict["validation_steps"],
										use_multiprocessing=True,
										initial_epoch=initial_epoch)

			# Merge these new results with existing results for previous batches:
			if results is not None:
				# Merge the two results lists:
				for key in results.history:
					results.history[key].extend(results_new.history[key])
			else:
				results = results_new

			initial_epoch = final_epoch
			
	runsec = timer.toc()
	
	# Create a new row for these results:
	new_res = pd.Series()
	assign_run_key(new_res, param_dict, run_started)
	assign_opt_key(new_res, opt_key)
	train_acc, val_acc = assign_results_history(new_res, model_key, runsec, results)
	# Add this to the results dataframe:
	try:
		fma_results = ut.load_obj(param_dict["fma_results_name"])
	except:
		fma_results = pd.DataFrame(dtype=float, columns = RESULTS_COLS)
	fma_results = fma_results.append(new_res, ignore_index = True)
	# And save:
	ut.save_obj(fma_results, param_dict["fma_results_name"])

	print(("\n{} for {} to yield {:0.1%} training accuracy "
		   "and {:0.1%} validation accuracy in {:d} \nepochs "
		   "(x{} training phases).").format(timer.time_from_sec(runsec),
										   param_dict["model_names"][model_key], 
										   train_acc,val_acc,
										   param_dict["pass_epochs"],
										   len(freeze_to) + 1))
	
	# Save trained weights:
	if save_weights:
		weights_save_name = os.path.join("saved_weights",
										 "{}_{}_{}_{}.h5".format(
												 model_key,
												 formatted(opt_key), # 4 elements
												 formatted(run_key(param_dict, 
												 				   run_started)),
												 timer.datetimepath()))
		ut.ensure_dir(weights_save_name)
		models[model_key].save_weights(weights_save_name)

	print("\n{} run complete at {}.".format(param_dict["model_names"][model_key], 
											timer.datetimestamp()))

	# Tell keras to clear the the tensorflow backend session (helps with memory leaks;
	# 		see: http://bit.ly/2xJZbAt )
	print("Clearing keras's backend Tensorflow session...\n")
	K.clear_session()






def run_fcnn_model(param_dict, generators, opts, opt_key, models, eta, 
				   pandas_save = True, save_weights = True,
				   epoch_batch_size = 10):
	assert(epoch_batch_size >= 1)

	model_key = "fcnn"
	if (param_dict["run_verbosity"] > 0):
			print("Using hidden size {} and optimizer {}...".format(
														param_dict["hidden_size"],
														opt_key))

	# Set up the input to accept FMA images:
	inp = keras.layers.Input(shape=(256,256,3,))
	
	# Add a flatten layer to make the input play nicely with these non-convolutional 
	# layers:
	x = keras.layers.Flatten()(inp)
			
	# Add a Flatten/Affine/BatchNorm/ReLU/Dropout/Affine-softmax-categorization block:
	predict = stack_two_layer_block(param_dict, x)

	# Construct the model:
	models[model_key] = keras.models.Model(inp,predict)

	# Compile the model
	models[model_key].compile(optimizer = opts[opt_key],
							  **param_dict["compile_args"])

	fcnn_pass_epochs = param_dict["pass_epochs"]*6 # Because all the other networks   
												   # train in multiple passes
	# Train the model:
	timer.tic()
	run_started = timer.datetimestamp()
	if (param_dict["run_verbosity"] > 0):
		print(("Fully connected network run begun at {}."
		   "\n\t[{} epochs on {} FMA on {} takes"
		   "\n\t{}.]\n").format(run_started, 
		   						   fcnn_pass_epochs, param_dict["which_size"],
		   						   param_dict["spu"].upper(),
		   						   eta))
                    
	initial_epoch = 0
	results = None
	# This loop of multiple checkpoints helps with memory management, which is 
	# 	  probably not necessary for FCNN but is included just in case - see also
	#     http://bit.ly/2hDHJay for more information.
	while initial_epoch < fcnn_pass_epochs:
		# Split into "epoch_batch_size"-epoch training batches
		final_epoch = min(initial_epoch+epoch_batch_size,
						  fcnn_pass_epochs)
		if (param_dict["run_verbosity"] > 0):
			print("\nTraining for epochs {} to {}...".format(initial_epoch+1, 
															 final_epoch))
		results_new = models[model_key].fit_generator(
									generators["train"],
									validation_data=generators["val"],
									verbose=param_dict["run_verbosity"], 
									epochs=final_epoch,
									steps_per_epoch=param_dict["steps_per_epoch"], 
									validation_steps=param_dict["validation_steps"],
									use_multiprocessing=True,
									initial_epoch=initial_epoch)

		# Merge these new results with existing results for previous batches:
		if results is not None:
			# Merge the two results lists:
			for key in results.history:
				results.history[key].extend(results_new.history[key])
		else:
			results = results_new

		# Now start from where we stopped on this round
		initial_epoch = final_epoch

	# results = models[model_key].fit_generator(
	# 								generators["train"],
	# 								validation_data=generators["val"],
	# 								verbose=param_dict["run_verbosity"], 
	# 								epochs=fcnn_pass_epochs,
	# 								steps_per_epoch=param_dict["steps_per_epoch"], 
	# 								validation_steps=param_dict["validation_steps"],
	# 								max_queue_size=10, # Default: 10
	# 								workers=1, # Default: 1
	# 								use_multiprocessing=True, # Default: False
	# 								) 

	runsec = timer.toc()
	
	# Create a new row for these results:
	if (pandas_save):
		new_res = pd.Series()
		assign_run_key(new_res, param_dict, run_started)
		assign_opt_key(new_res, opt_key)
		train_acc, val_acc = assign_results_history(new_res, model_key, runsec, 
													results)
		# Add this to the results dataframe:
		try:
				fma_results = ut.load_obj(param_dict["fma_results_name"])
		except:
				fma_results = pd.DataFrame(dtype=float, columns = RESULTS_COLS)
		fma_results = fma_results.append(new_res, ignore_index = "True")
		# And save:
		ut.save_obj(fma_results, param_dict["fma_results_name"])
	else:
		train_acc = results.history["categorical_accuracy"][-1]
		val_acc = results.history["val_categorical_accuracy"][-1]

	if (param_dict["run_verbosity"] > 0):
		print(("\n{} for {} to yield {:0.1%} training accuracy "
			   "and {:0.1%} validation accuracy in {:d} \nepochs "
			   "(x3 training phases).").format(timer.time_from_sec(runsec),
											   param_dict["model_names"][model_key], 
											   train_acc,val_acc,
											   param_dict["pass_epochs"]))
	
	# Save trained weights:
	if save_weights:
		weights_save_name = os.path.join("saved_weights",
										 "{}_{}_{}_{}.h5".format(
													 model_key,
													 formatted(opt_key), # 4 elements
													 formatted(run_key(param_dict, 
													 				   run_started)),
													 timer.datetimepath()))
		ut.ensure_dir(weights_save_name)
		models[model_key].save_weights(weights_save_name)
	
	if (param_dict["run_verbosity"] > 0):
		print("\nFully connected run complete at {}.".format(timer.datetimestamp()))
	

	# Tell keras to clear the the tensorflow backend session (helps with memory leaks;
	# 		see: http://bit.ly/2xJZbAt )
	if param_dict["run_verbosity"] > 0: # i.e. this is not a short-running/crossval run--
										# can't reset during crossval because tensorflow
										# will get cross about the optimizer being 
										# created on a different graph...
		print("Clearing keras's backend Tensorflow session...\n")
		K.clear_session()

	if (pandas_save):
		return new_res





def run_crossval(extern_params, opts, models):
	# Estimate time based on overridden parameters:
	num_opts = len(opts)
	est_sec = 60           # per optimizer; experimentally determined 
	print(("Cross-validation of {:d} optimizers with"
		   " manual parameters takes about {}.\n").format(
												num_opts,
												timer.time_from_sec(num_opts*est_sec)))

	# Configure for a few fast, GPU-utilizing epochs (changing the batch size requires 
	# unique generators, which is why this function doesn't take the generators as an 
	# argument):
	param_dict = extern_params.copy()
	param_dict["pass_epochs"] = 1
	param_dict["run_verbosity"] = 0
	if (param_dict["spu"]  == "gpu"): # Try to rely on the GPU if it's available
		param_dict["batch_size"] = 2048
		param_dict["steps_per_epoch"] = math.ceil(param_dict["dataset_size"]/
												  param_dict["batch_size"])
		param_dict["validation_steps"] = math.ceil(param_dict["dataset_size"]/
												   (8*param_dict["batch_size"]))
	else:
		param_dict["batch_size"] = 4
		param_dict["steps_per_epoch"] = 1
		param_dict["validation_steps"] = math.ceil(16/param_dict["batch_size"]) 
                                              # So we'll check at least 16 examples 
                                              # in validation--much smaller than 
                                              # this confuses Keras.

	param_dict["hidden_size"] = 8 # Use a very small network for cross-validation to
								  # speed up computation

	# Get generators with the appropriate batch size
	generators = {}
	(generators["train"], 
	 generators["val"], 
	 generators["test"]) = set_up_generators(param_dict)
	print()

	# Run the cross-validation
	best_val = -1
	crossval_results = pd.DataFrame(dtype=float, columns = RESULTS_COLS)
	best_opt_key = None
	count = 0

	# If we don't have a GPU available, this is just a code test run--finish early:
	if param_dict["spu"]  != "gpu":
		small_keys = random.sample(list(opts), 3)
		opts_new = {}
		for key in small_keys:
			opts_new[key] = opts[key]
		opts = opts_new

	print("Starting run at {}...".format(timer.datetimestamp()), end = "")
	for opt_key in opts:
		print("{:d}/{:d} ({:0.1%})".format(count+1, 
										   num_opts, 
										   count/num_opts), end = "")
		if (count < len(opts)-1):
			print("...", end="")

		row = run_fcnn_model(param_dict, generators,
							 opts, opt_key, {}, # Discard models in crossval
							 "4 min", save_weights = False)
		if row["Final Validation Accuracy"] > best_val:
			best_val = row["Final Validation Accuracy"]
			best_opt_key = opt_key
		
		crossval_results = crossval_results.append(row, ignore_index=True)

		count += 1

	print("\nBest optimizer: {} - {:0.1%} validation accuracy!".format(best_opt_key, 
																	   best_val))
	ut.save_obj(crossval_results, param_dict["crossval_results_name"])

	return best_opt_key
