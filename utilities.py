# File: utilities.py
# Author: Rosy Davis, rosydavis@ieee.org
# Last modified: 2017 Nov. 28
#
# A handful of useful utilities for debugging display, file system management, and
# saving and loading data.



import pickle
import os.path

# Save everything to a centralized location:
saved_obj_pathname = "saved_objects/"




# Utility for nested printing; starts a line with a specified number of tabs.
def print_tabs(tabs):
    for i in range(tabs):
        print("   ",end="")

# Print utility for dictionaries of lists and dictionaries; prints up to 10 layers deep
# cleanly. Useful for debugging.
def print_multilayer(name, data, tabs = 0):
    if (tabs > 10):
        print("OVER RECURSION LIMIT")
        raise
        
    print_tabs(tabs)
    print("'{}': {}".format(name, type(data)), end="")
    try:
        if (isinstance(data,dict) or isinstance(data,list)):
            print()
            for item in data:
                print_multilayer("{}".format(item), data[item], tabs=tabs+1)
        elif (isinstance(data,tuple)):
            print()
            for item, value in enumerate(data):
                try:
                    print_multilayer("{}".format(item), value, tabs=tabs+1)
                except:
                    print(" = {}".format(data))
        else:
            print(" = {}".format(data))
    except:
        print_tabs(tabs)
        print("'{}': {}".format(name,data))





# Makes sure that the directory chain for the specified path exists:
def ensure_dir(pathname):
    directory = os.path.dirname(pathname)
    if not os.path.exists(directory):
        os.makedirs(directory)



# Shortcuts for saving and loading pickleable objects; see
# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

# Save method. Ex:
#       save_obj(foo, "foo")
# will save variable foo to "saved_objects/foo.pkl".
def save_obj(obj, name, pathname = saved_obj_pathname):
    ensure_dir(pathname)
    with open(os.path.join(pathname, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load method. Ex:
#       foo = load_obj("foo")
# will load "saved_objects/foo.pkl".
def load_obj(name, pathname = saved_obj_pathname):
    with open(os.path.join(pathname, name + '.pkl'), 'rb') as f:
        return pickle.load(f)