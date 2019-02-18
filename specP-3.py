#!/usr/bin/env python3

# make a pdf plot file 
# input: qdp from xspec (pl d)

# 2018-06-26 cp from flux33-lu2sin.py
# 2018-08-28 cp from sig1.py
# 2018-10-15 cp from qdp2spe1.py
# 2018-10-15 cp from specP.py

import math
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

my_name = 'specP.py'

# set parameters

#fsize = (10,1.7)
fsize = (10,7)
print ('! create by ', my_name)

# inputs (1) data
outpdf = sys.argv[1]
in1 = sys.argv[2]
i=3
xr1,xr2 = float(sys.argv[i]),float(sys.argv[i+1])
yr1,yr2 = float(sys.argv[i+2]),float(sys.argv[i+3])

xrange = (xr1,xr2)
p1_yrange = (yr1,yr2)

print ('read (1) data...', in1, '\n')

# read data (1)

data = pd.read_csv(in1, sep='\s+',index_col=False,header=None,skiprows=3)

x = data[0] 
x_err = data[1]
y = data[2]
y_err = data[3]
m1,m2,m3 = data[4],data[5],data[6]

# significance
# cp from sig1.py
in1=sys.argv[i+4]
in2=sys.argv[i+5]
in3=sys.argv[i+6]
yr1,yr2 = float(sys.argv[i+7]),float(sys.argv[i+8])

p2_yrange = (yr1,yr2)

print ("read (2) \n")
print (in1, '\n', in2, '\n', in3)

print ("xrange=", xrange)
print ("p1_yrange=", p1_yrange)
print ("p2_yrange=", p2_yrange)

# significance, p2
names = ['x', 'y', 'c3', 'c4', 'c5','c6']
data = pd.read_csv(in1, sep='\s+',index_col=False,header=None)
p2_x1 = data[0]/1000.0 # keV
p2_y1 = data[1]

data = pd.read_csv(in2, sep='\s+',index_col=False,header=None)
p2_x2 = data[0]/1000.0 # keV
p2_y2 = data[1]

data = pd.read_csv(in3, sep='\s+',index_col=False,header=None)
p2_x3 = data[0]/1000.0 # keV
p2_y3 = data[1]

# plot (1), data
fig = plt.figure(figsize=fsize)

gs = gridspec.GridSpec(3, 1,height_ratios=[2, 1,1]
#gs = gridspec.GridSpec(2, 1,height_ratios=[2, 1]
                       )
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

# ax1 = fig.add_subplot(311)
col=['0.0','r']

ax1.errorbar(x, y, yerr=y_err, xerr=x_err, label="counts",color=col[0],fmt="none") # no line

# log Y plot
#y_err_log = np.log10(y+y_err) - np.log10(y)
# m1 = np.log10(m1)
# m2 = np.log10(m2)
# m3 = np.log10(m3)
#ax1.errorbar(x, np.log10(y), yerr=y_err_log, xerr=x_err, label="counts",color=col[0],fmt="none") # no line

ds_model = 'steps'

ax1.plot(x, m1,label="total model",color='0.0',drawstyle=ds_model)
ax1.plot(x, m2,label="c1",color='0.5',drawstyle=ds_model)
ax1.plot(x, m3,label="c2",color='0.5',drawstyle=ds_model)


ax1.set_yscale("log", nonposy='clip')

# https://qiita.com/Fortinbras/items/50500423888ef21429be
from matplotlib.ticker import ScalarFormatter
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#ax1.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
ax1.ticklabel_format(style="sci",  axis="y",scilimits=(-4,4))

#ax1.set(ylabel='Log(Counts /s/keV)')
ax1.set(ylabel='Counts /s/keV')

ax1.set_xlim(xrange)
ax1.set_ylim(p1_yrange)

#ax1.tick_params(direction='in',axis='x',labeltop=True) # 1.9,2.0 at top
ax1.tick_params(direction='in',axis='x', labelbottom=False)
ax1.tick_params(direction='in',axis='y')
ax1.grid()


# plot (2) 
# ax2 = fig.add_subplot(312)
col=['0.0','r']
ax2.plot(p2_x1, p2_y1,label="",drawstyle='steps-post',color=col[0])
ax2.plot(p2_x2, p2_y2, label="", color='r',drawstyle='steps-post')
ax2.plot(p2_x3, p2_y3, label="", color='b',drawstyle='steps-post')
ax2.set(ylabel='Sigma')
ax2.set_xlim(xrange)
ax2.set_ylim(p2_yrange)
ax2.grid()

for v in [-5,-3,0,3,5]:
        ax2.axhline(y=v,linewidth=0.5, color='b')

fig.subplots_adjust(wspace=0.0, hspace=0.0)

# model from model1.py
import mod_model as mm

redshift = 0.0173

# set parameters
# x1,x2 = float(sys.argv[1]),float(sys.argv[2])
i=12
y1,y2 = float(sys.argv[i]),float(sys.argv[i+1])
id_min, id_max = int(sys.argv[i+2]), int(sys.argv[i+3])
x1,x2 = xr1*(1.0+redshift),xr2*(1.0+redshift)
xrange = (x1,x2)
yrange = (y1,y2)
dir = "~/perseus_hitomi/spex-20161027/model-20170704/"

num_s = 30
ion = [' ' for i in range(num_s)]

# Al
i=0
c='0.0'
ion[i] = mm.MyIon(i, 'Al', 13, 12, c)
ion[i+1] = mm.MyIon(i+1, 'Al', 13, 13, c)

i=2; c='r'; name='Si'; Z=14
ion[i] = mm.MyIon(i, name, Z, 13, c)
ion[i+1] = mm.MyIon(i, name, Z, 14, c)

i=4; c='g'; name='P'; Z=15
ion[i] = mm.MyIon(i, name, Z, 14, c)
ion[i+1] = mm.MyIon(i, name, Z, 15, c)

#ele,n1,n2 = 'si', 13, 14
#ele,n1,n2 = 'p', 14, 15

i=6; c='b'; name='S'; Z=16
ion[i] = mm.MyIon(i, name, Z, 15, c)
ion[i+1] = mm.MyIon(i, name, Z, 16, c)

i=8; c='c'; name='Cl'; Z=17
ion[i] = mm.MyIon(i, name, Z, 16, c)
ion[i+1] = mm.MyIon(i, name, Z, 17, c)

i=10; c='m'; name='Ar'; Z=18
ion[i] = mm.MyIon(i, name, Z, 17, c)
ion[i+1] = mm.MyIon(i, name, Z, 18, c)

i=12; c='0.0'; name='K'; Z=19
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i, name, Z, Z, c)

i=14; c='r'; name='Ca'; Z=20
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i, name, Z, Z, c)

i=16; c='g'; name='Ti'; Z=22
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i, name, Z, Z, c)

i=18; c='b'; name='Cr'; Z=24
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i, name, Z, Z, c)

i=20; c='c'; name='Mn'; Z=25
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i, name, Z, Z, c)

i=22; c='0.0'; name='Fe'; Z=26
# c='g' NG
c='m' 
ion[i] = mm.MyIon(i, name, Z, Z-3, c)
ion[i+1] = mm.MyIon(i+1, name, Z, Z-2, c)
# H, He-like
c='0.0'
ion[i+2] = mm.MyIon(i+2, name, Z, Z-1, c)
ion[i+3] = mm.MyIon(i+3, name, Z, Z, c)

# Ni
i=26; c='r'; name='Ni'; Z=28
ion[i] = mm.MyIon(i, name, Z, Z-1, c)
ion[i+1] = mm.MyIon(i+1, name, Z, Z, c)

dstyle = 'steps-post'

font_s = 16
for i in range(id_min,id_max+1):
        print (i, ion[i].name)
        in1 = os.path.join(dir,ion[i].fname)
        print ("in1=", in1)
        try:
                print ("reading", in1)
                data = pd.read_csv(in1, sep='\s+',index_col=False,header=None,skiprows=4)
                data.columns=['nr','ene','cont','line','total']
                x = data['ene'] # OK
                y = np.log10(data['line'])
                ax3.plot(x, y,drawstyle=dstyle,color=ion[i].col,linestyle=ion[i].linestyle)
                #                ax.fill(x, y,ion[i].col)
                if ((i % 2) == 0):
                        xp = 0.05 + 0.025*i
                        yp = 0.9
                        text = ion[i].name
                        #                        print (i, ion[i].name,xp,yp)
                        ax3.text(xp, yp, text, color=ion[i].col, transform=ax3.transAxes,fontsize=font_s)
                        ax3.text(xp, 0.80, str(ion[i].atom_num), color=ion[i].col, transform=ax3.transAxes,fontsize=font_s)
        except:
                print ("some error...")
ax3.set_xlim(xrange)
ax3.set_ylim(yrange)
ax3.set(ylabel='Log(Lx)')
ax3.set(xlabel='Rest-frame Energy (keV)')
ax3.grid()
ax3.tick_params(direction='in')

# make pdf
pd=0.0
fig.savefig(outpdf,dpi=600,bbox_inches='tight',pad_inches=0.0) # latex ?
#fig.savefig(outpdf,dpi=600)
print ("check %s " % outpdf)
