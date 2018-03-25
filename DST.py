import urllib2
import sys
from BeautifulSoup import BeautifulSoup
import datetime
from calendar import monthrange
import matplotlib.pyplot as plt
import matplotlib as mpl
import sched, time
import numpy as np
import threading
from GUI import GUI
from PredictionModel import PredictionModel

class DST:

    def __init__(self):

        #Load GUI class
        self.G = GUI()
        #Load Prediction Model
        self.PM = PredictionModel()

        #Prepare the Figure
        plt.ion()
        self.fig = mpl.figure.Figure(figsize = (10,5),dpi=80, facecolor='w',)
        self.ax = self.fig.add_subplot(111)

        #Urls for Real-Time DST and IMF data
        self.url_dst = 'http://wdc.kugi.kyoto-u.ac.jp/dst_realtime/presentmonth/index.html'
        self.url_imf = 'https://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2018.dat'

        #Load data
        self.fill_data()
        #Plot data
        self.plot_data()

        #Intilialize a scheduler
        self.scheduler = sched.scheduler(time.time, time.sleep)

        #Load new values
        self.update()
        #Show GUI
        self.G.start_gui()


    def fill_data(self):
        #Initializer of data arrays, creates empty data arrays for DST and IMF

        #Get current date and month length
        month = datetime.datetime.utcnow().month
        year = datetime.datetime.utcnow().year
        self.length = monthrange(year, month)[1]

        #Create arrays for dst and imf with required length
        self.data_dst = np.zeros((self.length, 24), )
        self.data_imf = np.zeros((self.length, 24), )

        #current time
        today = datetime.datetime.utcnow().day
        hour = datetime.datetime.utcnow().hour

        if(hour == 0):
            hour = 24
            today = today -1
            if(today == 0):
                return True

        #Latest time of update
        self.today = today
        self.hour = hour

        #Load available data
        self.update_data(1, 0, today, hour - 1)

    def update_data(self,from_day, from_time, to_day, to_time):
        self.update_dst(from_day, from_time, to_day, to_time)
        self.update_imf(from_day, from_time, to_day, to_time)

    def update_dst(self, from_day, from_time, to_day, to_time):

        #Load the page from Real-time DST data
        response = urllib2.urlopen(self.url_dst)
        content = response.read()
        parsed_html = BeautifulSoup(content)
        #Get pre tag containing the data
        data = str(parsed_html.body.find('pre', attrs={'class': 'data'}).text).splitlines()

        #get data day be day
        for i in xrange(from_day,to_day+1):

            line = data[5+i+(i-1)/5]
            for j in xrange(0,24):
                #for each day get valid hours
                if(i == from_day and j < from_time):
                    continue
                if (i == to_day and j > to_time):
                    break

                #Parse dst
                value = float(line[(j+1) * 4+(j/8)-1: (j+2) * 4+(j/8)-1])

                self.data_dst[i - 1, j] = value

        return True

    def update_imf(self, from_day, from_time, to_day, to_time):
        #Same as DST function
        response = urllib2.urlopen(self.url_imf)
        content = response.read()
        data = str(content).splitlines()

        for i in xrange(from_day,to_day+1):
            line = data[(i-1)*24:i*24]
            for j in xrange(0,24):
                if(i == from_day and j < from_time):
                    continue
                if (i == to_day and j > to_time):
                    break
                splitted = line[j].split()
                value = float(splitted[14])

                self.data_imf[i - 1, j] = value

        return True

    def plot_data(self):
        #Redraw the figure
        self.ax.clear()
        #Flatten the data
        data_dst = (np.ndarray.flatten(self.data_dst))[0:(self.today - 1) * 24 + self.hour]
        data_imf = (np.ndarray.flatten(self.data_imf))[0:(self.today - 1) * 24 + self.hour]

        #Plot real-time DST and IMF (Bz,GSE)
        self.ax.plot(np.arange(1,(len(data_dst)-2)/23.0,1.0/24.0),data_dst, 'b', label = 'DST')
        self.ax.plot(np.arange(1,(len(data_imf)-2)/23.0,1.0/24.0),data_imf, 'g', label = 'IMF (Bz, GSE)')

        #If have enough data predict next DST for future 5 hours
        if(len(data_dst)>4):
            #Prepare input data for the model
            p_input = np.append(data_dst[-5:], data_imf[-5:])
            #Make predictions
            p_dst = self.PM.predict(p_input)
            #Plot predictions
            self.ax.plot(np.arange((len(data_imf) - 2) / 23.0, ((len(data_imf) - 2) / 23.0) + 5.0 / 24, 1.0 / 24.0),
                         p_dst,'r', label = 'Predicted DST')

        self.ax.set_title("DST and IMF data (updated: "+ str(datetime.datetime.utcnow())+")")
        self.ax.set_ylabel('Value (nT)')
        self.ax.set_xlabel('Day of the month')
        self.ax.legend()
        self.ax.set_xlim(1,self.length)

        try:
            self.G.draw(self.fig)
        except RuntimeError:
            pass


    def check_running(self):
        #Check whether the user closed the program
        if(not self.G.is_active()):
            #If yes, kill the background thread
            sys.exit(0)
        else:
            #If no check if it's time to update the data (once in an hour at 15 minutes)
            if((15 - datetime.datetime.utcnow().minute)==0):
                self.fill_data()
                self.plot_data()

    def update(self):
        #Background thread for data updates
        if(self.G.active):
            #Check every minute if program running and update data once in an hour
            self.scheduler.enter(60, 2, self.update, {})
            self.scheduler.enter(60, 1, self.check_running,{})
            th = threading.Thread(target=self.scheduler.run,args={})
            th.start()
        else:
            sys.exit(0)
