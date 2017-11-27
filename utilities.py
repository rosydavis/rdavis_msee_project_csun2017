import pickle
import os.path

def print_tabs(tabs):
    for i in range(tabs):
        print("   ",end="")

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

def ensure_dir(pathname):
    directory = os.path.dirname(pathname)
    # Make sure the directory chain exists
    if not os.path.exists(directory):
        os.makedirs(directory)

# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
saved_obj_pathname = "saved_objects/"

def save_obj(obj, name, pathname = saved_obj_pathname):
    ensure_dir(pathname)
    with open(os.path.join(pathname, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, pathname = saved_obj_pathname):
    with open(os.path.join(pathname, name + '.pkl'), 'rb') as f:
        return pickle.load(f)