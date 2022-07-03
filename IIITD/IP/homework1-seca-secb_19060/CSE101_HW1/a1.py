#Name - Bhavay Aggarwal	
#Roll Number - 2018384
#Section - B
#Group - 1

import urllib.request
import datetime
# function to get weather response
def weather_response(location, API_key):
	a1 = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q='+location+'&APPID='+API_key )
	a2 = a1.read().decode("utf-8")
	return a2
	# write your code 

# function to check for valid response 
def has_error(location,json):
	x1 = json.index('"name":') +8
	y1 = json.index('","coord"')

	if json[x1:y1]!=location:
	    return True
	# write your code 
	
		

# function to get attributes on nth day
def get_temperature (json, n=0,t="03:00:00"):
	current_date = datetime.date.today()
	end_date = current_date + datetime.timedelta(days=int(n))
	date_time = str(end_date) + ' ' + str(t)
	num =0
	x2 = json.index('"temp"')+7
	y2 = json.index(',"temp_min"')
	x1 = json.index('"dt_txt":') +10
	y1 = x1+19
	def find(x1,y1,x2,y2):
	    if json[x1:y1]==date_time:
		    return json[x2:y2]
		elif x1> len(json):
			return ""
	    else:
		    x2 = json.index('"temp"',x2+1)+7
		    y2 = json.index(',"temp_min":',y2+1)
		    x1 = json.index('"dt_txt":',x1+1) +10
		    y1 = x1+19
		    find(x1,y1,x2,y2)
	 
	# write your code 


def get_humidity(json, n=0,t="3:00:00"):
	current_date = datetime.date.today()
	end_date = current_date + datetime.timedelta(days=int(n))
	date_time = str(end_date) + ' ' + str(t)
	num =0
	x2 = json.index('"humidity":')+11
	y2 = json.index(',"temp_kf"')
	x1 = json.index('"dt_txt":') +10
	y1 = x1+19
	def find(x1,y1,x2,y2):
	    if json[x1:y1]==date_time:
		    return json[x2:y2]
		    num=1
	    elif x1> len(json):
			return ""
	    else:
		    x2 = json.index('"humidity":',x2+1)+11
		    y2 = json.index(',"temp_kf"',y2+1)
		    x1 = json.index('"dt_txt":',x1+1) +10
		    y1 = x1+19
		    find(x1,y1,x2,y2)
	# write your code 

def get_pressure(json, n=0,t="03:00:00"):
	current_date = datetime.date.today()
	end_date = current_date + datetime.timedelta(days=int(n))
	date_time = str(end_date) + ' ' + str(t)
	num =0
	x2 = json.index('"pressure"')+11
	y2 = json.index(',"sea_level"')
	x1 = json.index('"dt_txt":') +10
	y1 = x1+19
	def find(x1,y1,x2,y2):
	    if json[x1:y1]==date_time:
		    return json[x2:y2]
		    num=1
	    elif x1> len(json):
			return ""
	    else:
		    x2 = json.index('"pressure"',x2+1)+11
		    y2 = json.index(',"sea_level"',y2+1)
		    x1 = json.index('"dt_txt":',x1+1) +10
		    y1 = x1+19
		    find(x1,y1,x2,y2)
		 
	# write your code 
	 

def get_wind(json, n=0,t="03:00:00"):
	current_date = datetime.date.today()
	end_date = current_date + datetime.timedelta(days=int(n))
	num =0
	date_time = str(end_date) + ' ' + str(t)
	x2 = json.index('"speed":')+8
	y2 = json.index(',"deg":')
	x1 = json.index('"dt_txt":') +10
	y1 = x1+19
	def find(x1,y1,x2,y2):
	    if json[x1:y1]==date_time:
		    return json[x2:y2]
		    num=1
	    elif x1> len(json):
			return ""
	    else:
		    x2 = json.index('"speed":',x2+1)+8
		    y2 = json.index(',"deg":',y2+1)
		    x1 = json.index('"dt_txt":',x1+1) +10
		    y1 = x1+19
		    find(x1,y1,x2,y2)
	# write your code 
	

def get_sealevel(json, n=0,t="03:00:00"):
	current_date = datetime.date.today()
	end_date = current_date + datetime.timedelta(days=int(n))
	date_time = str(end_date) + ' ' + str(t)
	num =0
	x2 = json.index('"sea_level":')+12
	y2 = json.index(',"grnd_level"')
	x1 = json.index('"dt_txt":') +10
	y1 = x1+19
	while num==0 and x1<len(json):
	    if json[x1:y1]==date_time:
		    return json[x2:y2]
		    num=1
	    elif x1> len(json):
			return ""
	    else:
		    x2 = json.index('"sea_level":',x2+1)+12
		    y2 = json.index(',"grnd_level"',y2+1)
		    x1 = json.index('"dt_txt":',x1+1) +10
		    y1 = x1+19
		    find(x1,y1,x2,y2)
	# write your code




