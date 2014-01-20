'''
Created on 20/01/2014

@author: olena
'''
import numpy as np
import matplotlib.pyplot as plt


COLORS =["#8533D6","#5C5C8A","#a36e81","#7ba29a","#6600FF","#5C85D6","#006600","#1963D1","#0066FF","#5C5C8A","#6666FF",]
light_COLORS = ["#FFFF99","#66CCFF","#7f7f7f","#B8E65C","#52A3CC",]
Q_COLORS = ["#FFFF99","#1963D1","#0066FF","#B8E65C",]
short_color = "#FF3300"
long_color = "#009900"
STR_COLORS = ["#3300FF","#FF6600","#585858","#a36e81","#006600","#FF0000","#FF3399","#6600FF","#8533D6","#7ba29a","#5C5C8A",]

#############################################
## auxiliary functions
#############################################

def format_legend(leg):
    try:
        for text in leg.get_texts():
            text.set_fontsize('x-small')
        for line in leg.get_lines():
            line.set_linewidth(0.7)
    except:
        pass

def add_to_scatter_plots(ax,xlabel,title):
    ax.axhline(color='#000000')
    ax.axvline(color='#000000')
    ax.set_xlabel(xlabel,fontsize=9)
    ax.tick_params(labelsize=9)
    ax.set_title(title,fontsize=10,color='blue')
    ax.grid(True)
    t_leg = ax.legend(loc='best')
    format_legend(t_leg)

def form_suptitle(suptitle,date1=None, date2=None):
    if date1 != None:
        suptitle += " for "+date1.strftime("%d/%m/%Y")
    if date2 != None:
        suptitle += " thru "+ date2.strftime("%d/%m/%Y")
    return suptitle


def format_plot(ax,x_label,title=None,y_label = None,use_legends=True):
    ax.set_xlabel(x_label,fontsize=9)
    ax.axhline(color='#000000')
    if use_legends:
        t_leg = ax.legend(loc='best')
        format_legend(t_leg)
    ax.tick_params(labelsize=9)
    if title != None:
        ax.set_title(title,color='blue',fontsize=8)
    if y_label != None:
        ax.set_ylabel(y_label,fontsize=9)
        
def simple_scatter_plot(x,y,xlab=None,ylab=None,title=None,fn=None,out_dir=""):
    fig = plt.figure(facecolor='w', edgecolor='b', frameon=True)#,axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.25,wspace=0.35,bottom=0.2)  
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y,c=COLORS[1])
    if xlab != None:
        ax.set_xlabel(xlab)
    if ylab != None:
        ax.set_ylabel(ylab)
    if title !=None:
        fig.suptitle(title)
    if fn !=None:
        plt.savefig(out_dir+fn+".jpg")
    else:
        plt.show()
       