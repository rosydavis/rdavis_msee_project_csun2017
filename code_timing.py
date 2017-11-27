# For code timing and estimation:
from timeit import default_timer as timer
from datetime import datetime  
from datetime import timedelta

settings = {}
settings["dateform"] = "%A, %Y %B %-d, %-I:%M %p" # The "%-d" drops the leading 0 on day
settings["pathform"] = "%Y-%m-%d+%H%M" # but we want the leading 0 here
settings["verbose"] = False # Not really used right now, but just in case

timers = {}

''' Given a time duration in seconds, converts to a string in "HH:MM:SS" format, or,
	if s < 60, a nicely-formatted decimal value with 'sec' at the end. '''
def time_from_sec(s):
	if (s < 60): # Less than a minute; just print nicely
		return "{:0.3f} sec".format(s)
	s = (int)(round(s,0))
	m, s = divmod(s, 60)
	h, m = divmod(m, 60)
	return "{:02d}:{:02d}:{:02d}".format(h,m,s)

''' Returns a string representation ofthe current date and time in the format specified in 
	settings["dateform"]. '''
def datetimestamp():
	return datetime.now().strftime(settings["dateform"])

''' Returns a string representation ofthe current date and time in the format specified in 
	settings["pathform"]. '''
def datetimepath():
	return datetime.now().strftime("%Y-%m-%d+%H%M")

''' Start or restart the timer called "name". '''
def tic(name = 'total'):
	timers[name] = timer()

''' Lap (i.e., get the time, but do not restart) the timer called "name". If the timer 
	has not been started, raises a NameError. '''
def toc(name = 'total'):
	if name not in timers:
		raise NameError("Timer '{:s}' has not been started.".format(name))
	return (timer() - timers[name])
