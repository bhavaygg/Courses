#Name - Bhavay Aggarwal	
#Roll Number - 2018384
#Section - B
#Group - 1

import unittest
import urllib.request
from a1 import weather_response
from a1 import has_error
from a1 import get_temperature 
from a1 import get_humidity
from a1 import get_pressure
from a1 import get_wind
from a1 import get_sealevel
#Storing contents of a webpage in a variable instead of copy pasting it over and over again
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=Delhi&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html=response.read().decode("utf-8")
#Reading data from Delhi forecast and storing in a variable
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=Mumbai&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html1=response.read().decode("utf-8")
#Reading data from Mumbai forecast and storing in a variable
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=Mumbai&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html1=response.read().decode("utf-8")
#Reading data from Mumbai forecast and storing in a variable
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=New%20York&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html2=response.read().decode("utf-8")
#Reading data from New York forecast and storing in a variable
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=Sydney&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html3=response.read().decode("utf-8")
#Reading data from Sydney forecast and storing in a variable
response = urllib.request.urlopen('http://api.openweathermap.org/data/2.5/forecast?q=Kolkata&APPID=b0c67b4c47bad5a7c79bf5401369860e')
html4=response.read().decode("utf-8")
#Reading data from Kolkata forecast and storing in a variable
class testpoint(unittest.TestCase):
	
    def test_weather_response(self):
        self.assertEqual(weather_response("Delhi","b0c67b4c47bad5a7c79bf5401369860e"),html)
        self.assertEqual(weather_response("Mumbai","b0c67b4c47bad5a7c79bf5401369860e"),html1)
        self.assertEqual(weather_response("New%20York","b0c67b4c47bad5a7c79bf5401369860e"),html2)
        self.assertEqual(weather_response("Sydney","b0c67b4c47bad5a7c79bf5401369860e"),html3)
        self.assertEqual(weather_response("Kolkata","b0c67b4c47bad5a7c79bf5401369860e"),html4)

    def test_has_error(self):
	    self.assertTrue(has_error("12", html))
	    self.assertTrue(has_error("Bombay", html1))
	    self.assertTrue(has_error("London", html2))
	    self.assertTrue(has_error("123", html3))
	    self.assertTrue(has_error("123555555", html3))
        #Incorrect city names so that the function always returns True

#The functions below return desired value for the date 10-09-2018, so slight modification in n will give the correct value   
    def test_get_temperature(self):
        self.assertAlmostEqual(float((get_temperature(html,'2',"06:00:00"))),302.94,delta=5)
        self.assertAlmostEqual(float((get_temperature(html1,'2',"06:00:00"))),299.44,delta=5)
        self.assertAlmostEqual(float((get_temperature(html2,'2',"06:00:00"))),291.84,delta=5)
        self.assertAlmostEqual(float((get_temperature(html3,'2',"06:00:00"))),291.29,delta=5)
        self.assertAlmostEqual(float((get_temperature(html4,'2',"06:00:00"))),303.54,delta=5) 	

    def test_get_humidity(self):
        self.assertAlmostEqual(int(get_humidity(html,'2',"06:00:00")),93,delta=5)
        self.assertAlmostEqual(int(get_humidity(html1,'2',"06:00:00")),100,delta=5)
        self.assertAlmostEqual(int(get_humidity(html2,'2',"06:00:00")),91,delta=5)
        self.assertAlmostEqual(int(get_humidity(html3,'2',"06:00:00")),71,delta=5)
        self.assertAlmostEqual(int(get_humidity(html4,'2',"06:00:00")),83,delta=5)

    def test_get_pressure(self):
    	self.assertAlmostEqual(float((get_pressure(html,'2',"06:00:00"))),998.28,delta=5)
    	self.assertAlmostEqual(float((get_pressure(html1,'2',"06:00:00"))),1025.19,delta=5)
    	self.assertAlmostEqual(float((get_pressure(html2,'2',"06:00:00"))),1031.75,delta=5)
    	self.assertAlmostEqual(float((get_pressure(html3,'2',"06:00:00"))),1032.89,delta=5)
    	self.assertAlmostEqual(float((get_pressure(html4,'2',"06:00:00"))),1022.27,delta=5)
	
    def test_get_wind(self):
	    self.assertAlmostEqual(float((get_wind(html,'2',"06:00:00"))),2.22,delta=5)
	    self.assertAlmostEqual(float((get_wind(html1,'2',"06:00:00"))),5.27,delta=5)
	    self.assertAlmostEqual(float((get_wind(html2,'2',"06:00:00"))),5.47,delta=5)
	    self.assertAlmostEqual(float((get_wind(html3,'2',"06:00:00"))),1.32,delta=5)
	    self.assertAlmostEqual(float((get_wind(html4,'2',"06:00:00"))),4.37,delta=5)

    def test_get_sealevel(self):
	    self.assertAlmostEqual(float((get_sealevel(html,'2',"06:00:00"))),1022.73,delta=5)
	    self.assertAlmostEqual(float((get_sealevel(html1,'2',"06:00:00"))),1025.77,delta=5)
	    self.assertAlmostEqual(float((get_sealevel(html2,'2',"06:00:00"))),1035.1,delta=5)
	    self.assertAlmostEqual(float((get_sealevel(html3,'2',"06:00:00"))),1039.03,delta=5)
	    self.assertAlmostEqual(float((get_sealevel(html4,'2',"06:00:00"))),1023.54,delta=5)


	
if __name__=='__main__':
	unittest.main()
