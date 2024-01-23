#!/usr/bin/env python

import sys
import os
import scipy.optimize as opt
#sys.path.append(r'C:\Users\sarah\Google Drive\WEAVE\Code\fstest')
#sys.path.append(r"C:\Users\sarah\Google Drive\WEAVE\Code\fstest\SarahPos")
import numpy as np
import scipy.linalg as linalg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splev, splrep

import scipy.optimize as opt
from scipy import interpolate

os.chdir(r"C:\Users\sarah\Google Drive\WEAVE\Code\fstest\SarahPos")

####defining global variables #########
guide_R = sorted(np.array([1007,2,125,128,251,254,377,380,503,506,629,632,755,758,881,884]))
spacing = 360./(168*2)
guide_theta = np.array([ 1.07142857, 45., 46.07142857, 90.,91.07142857, 135., 136.07142857, 180., 181.07142857, 225., 226.07142857, 270., 271.07142857, 315., 316.07142857, 360.])

#####defining the functions to convert retractor angle to XY positions for plate extrapolation ##################
f_y = lambda theta, r: r*np.cos((theta)*(np.pi/180.))
f_x =  lambda theta, r: r*np.sin(theta*(np.pi/180.))

class base_zmap():
    def __init__(self,filename):
        #plotting_methods()
        if type(filename)==str:
            self.zmapdata = np.loadtxt(filename)
        else:
            self.zmapdata = filename
        self.gridx,self.gridy=np.mgrid[-200.:200.:5,-200.:200.:5]
        self.XX = self.gridx.flatten()
        self.YY = self.gridy.flatten()

    def fit(self):
        self.A=np.c_[np.ones(self.zmapdata.shape[0]),self.zmapdata[:,:2],np.prod(self.zmapdata[:,:2],axis=1),self.zmapdata[:,:2]**2]
        self.C,_,_,_ = linalg.lstsq(self.A,self.zmapdata[:,2])

        self.gd = interpolate.griddata((self.zmapdata[1:,0], self.zmapdata[1:, 1]),self.zmapdata[1:,2], (self.gridx,self.gridy),'nearest').T
        self.Z=np.dot(np.c_[np.ones(self.XX.shape),self.XX,self.YY,self.XX*self.YY,self.XX**2,self.YY**2], self.C).reshape(self.gridx.shape).T
        self.extent = [np.min(self.XX) , np.max(self.XX), np.min(self.YY) , np.max(self.YY)]
        return

    def plot(self):
        
        plt.imshow(self.Z, extent=self.extent)
        plt.colorbar()
        plt.xlabel('X position [mm]')
        plt.ylabel('Y position [mm]')
        plt.show()
        return

    def zvalue(self,x,y):
        zval = np.dot(np.c_[np.ones(np.size(x)),x,y,x*y,x**2,y**2],self.C)
        return zval


class Zmap():
    def __init__(self,filename):
        if filename is not str:
            self.zmapdata = filename
        else:
            self.zmapdata = np.loadtxt(filename)
        self.title = filename
        subrange1 = np.where(self.zmapdata[:,3] == 0.)
        offset1 = self.zmapdata[0,2] - self.zmapdata[3,2]
        subrange2 = np.where(self.zmapdata[:,3] == -90.)
        offset2 = self.zmapdata[3,2] - self.zmapdata[3,2]
        subrange3 = np.where(self.zmapdata[:,3] == 90.)
        offset3 = self.zmapdata[1,2] - self.zmapdata[3,2]
        subrange4 = np.where(self.zmapdata[:,3] == -180.)
        offset4 = self.zmapdata[2,2] - self.zmapdata[3,2]
        self.zmapdata[subrange1,0] = self.zmapdata[subrange1,0] + 55
        self.zmapdata[subrange2,1] = self.zmapdata[subrange2,1] - 55
        self.zmapdata[subrange3,1] = self.zmapdata[subrange3,1] + 55
        self.zmapdata[subrange4,0] = self.zmapdata[subrange4,0] - 55
        self.zmapdata[subrange1,2] = self.zmapdata[subrange1,2] - offset1
        self.zmapdata[subrange2,2] = self.zmapdata[subrange2,2] - offset2
        self.zmapdata[subrange3,2] = self.zmapdata[subrange3,2] - offset3
        self.zmapdata[subrange4,2] = self.zmapdata[subrange4,2] - offset4
        self.gridx,self.gridy=np.mgrid[-205.:205.:5,205.:-205.:-5]
        self.XX = self.gridx.flatten()
        self.YY = self.gridy.flatten()
        return

    def fit(self):
        self.A=np.c_[np.ones(self.zmapdata.shape[0]),self.zmapdata[:,:2],np.prod(self.zmapdata[:,:2],axis=1),self.zmapdata[:,:2]**2]
        self.C,_,_,_ = linalg.lstsq(self.A,self.zmapdata[:,2])
        self.gd = interpolate.griddata((self.zmapdata[1:,0], self.zmapdata[1:, 1]),self.zmapdata[1:,2], (self.gridx,self.gridy),'nearest').T
        self.Z=np.dot(np.c_[np.ones(self.XX.shape),self.XX,self.YY,self.XX*self.YY,self.XX**2,self.YY**2], self.C).reshape(self.gridx.shape).T
        self.extent = [np.min(self.XX) , np.max(self.XX), np.min(self.YY) , np.max(self.YY)]
        return

    def plot(self,ax=plt):
        im=ax.imshow(self.Z,extent=self.extent)
        plt.colorbar(im,ax=ax)
        return

    def plotdiff(self,map,basemap,ax=plt):
        self.zmapdata.T[2] = map.T[2] - basemap.T[2]
        self.fit()
        im = ax.imshow(self.Z,extent=self.extent)
        patch = patches.Circle((0,0),radius=205,transform=ax.transData)
        im.set_clip_path(patch)
        plt.colorbar(im,ax=ax)
        return

    def zvalue(self,x,y):
        zval = np.dot(np.c_[np.ones(np.size(x)),x,y,x*y,x**2,y**2],self.C)
        return zval


class Parks(base_zmap, Zmap):

    def __init__(self, plate_map_file, guide_parks, robot, plate, dir_guides=None, dir_plate=None, base=False) -> None:
        self.base = base
        if dir_plate is not None:
            os.chdir(dir_plate)
        if base: 
            base_zmap.__init__(self, plate_map_file)
            base_zmap.fit(self)
            self.Z_C = self.C
        else:
            Zmap.__init__(self, plate_map_file)
            Zmap.fit(self)
            self.Z_C = self.C

        self.theta = np.arange(0, 360, 360/336)

        self._guide_parks = guide_parks
        self._plate_map_file = plate_map_file

        self._robot = robot
        self._plate = plate
        if plate_map_file is not str:
            self._plate_data = plate_map_file
        else:
            self._plate_data = np.loadtxt(plate_map_file)
        self.guide_data = guide_parks

        if type(guide_parks)==str:
            if dir_guides is not None:
                os.chdir(dir_guides)
            self.guide_data = np.loadtxt(guide_parks)
        self.guide_data = guide_parks[guide_parks[:,0].argsort()]

        print('Fits initiated')
        return

    def extrapolate(self):
        self.park_x = f_x(self.theta, 289.75)
        self.park_y = f_y(self.theta, 289.75)
        
        if self.base:
            self.plate_parks = base_zmap.zvalue(self,self.park_x, self.park_y)
            self.plate_centre = base_zmap.zvalue(self, 0., 0.)
        else:
            self.plate_parks = Zmap.zvalue(self,self.park_x, self.park_y)
            self.plate_centre = Zmap.zvalue(self, 0., 0.)
        return

    def find_complete_offsets(self):
        self._offsets = self.plate_parks - self._guide_parks[:,1]
        self.complete_offsets = np.vstack((self._guide_parks[:,0], self._offsets)).transpose()

        return self.complete_offsets
    

class Compare_parks(Parks):

    def __init__(self, ZD_40_plate, new_plate_map, ZD_40_parks, Guide_parks, dir_guides=None, dir_ZD_plate=None, dir_new_plate=None) -> None:
        #sorting the guide park ordering by retractor number to prevent confusion
        Guide_parks = Guide_parks[Guide_parks[:,0].argsort()]
        #Setting up the ZD40 base model
        self.ZD40 = Parks(plate_map_file=ZD_40_plate, guide_parks=ZD_40_parks, robot=0, plate='A', dir_guides=dir_guides, dir_plate=dir_ZD_plate)
        self.ZD40.extrapolate()
        self.base_plate_centre = self.ZD40.plate_centre
        self.ZD40.find_complete_offsets()

        #Defining the difference between the extrpolated plate map and the full set of measured parks
        self.delta_M = self.ZD40.complete_offsets
        self.f_40 = self.ZD40.plate_parks

        #isolating the offsets to just the guide values
        places = np.arange(2, 1008, 3)
        self.F_40 = np.vstack((places, self.f_40)).transpose()
        self.guides_40 = np.asarray([[item[0], item[1]] for index, item in enumerate(self.F_40) if item[0] in guide_R])
        self.guide_offsets = np.asarray([[item[0], item[1]] for index, item in enumerate(self.delta_M) if item[0] in guide_R])

        # defining new plate map and measured guide positions
        self.new_position = Parks(plate_map_file=new_plate_map, guide_parks=Guide_parks, robot=0, plate='A', dir_guides=dir_guides, dir_plate=dir_new_plate)
        self.new_position.extrapolate()
        self.new_plate_centre = self.new_position.plate_centre

        #turning into predicted values bassed on ZD40 offsets
        self.f_new = self.new_position.plate_parks
        self.M_predicted = self.f_new - self.delta_M[:,1]

        #Isolating the predictions to the guide retractors
        self.mm = np.vstack((places, self.M_predicted)).transpose()
        self.M_pre_guides = np.asarray([[item[0], item[1]] for index, item in enumerate(self.mm) if item[0] in guide_R])

        #defining the new measured parks as a class variable for future sanity checks
        self.measured_parks = Guide_parks

        return

    def plot_diffs(self, show=True):
        theta = np.arange(0, 360, 360/336)

        plt.plot(theta, self.M_predicted, label='Predicted park heights')            
        plt.scatter(guide_theta, self.measured_parks[:,1], label='Guides')
        plt.legend()

        if show:
                plt.show()
                return
        return


class residual_fit(Compare_parks):

    def __init__(self, base_plate, new_plate, base_parks, new_guides, G_dir, base_plate_dir, new_plate_dir) -> None:
        
        super().__init__(base_plate, new_plate, base_parks, new_guides, dir_guides=G_dir, dir_ZD_plate=base_plate_dir, dir_new_plate=new_plate_dir)
        self.theta = np.arange(0, 360, 360/336)

        self.guide_res = np.vstack(( self.measured_parks[:,0],self.measured_parks[:,1] - self.M_pre_guides[:,1])).transpose()
        return

            
    def spline_fitting(self, show=True, save=True, filename=None):
        spl = splrep(self.guide_res[:,0], self.guide_res[:,1])
        self.xx = np.arange(0, 1009, 1)#np.linspace(0, 1008, 1008).astype(int)
        self.yy = splev(self.xx, spl)

        self.all_park_res = np.vstack((self.xx, self.yy)).transpose
        if save and filename:
            assert type(filename)==str, 'Filename must be a string ending in .txt'
            np.savetxt(filename, self.all_park_res, fmt=('%d','%1.9f'))

        if show:
            plt.scatter(self.guide_res[:,0], self.guide_res[:,1])
            plt.plot(self.xx, self.yy)
            plt.show()

        return
