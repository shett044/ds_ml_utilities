import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import pandas as pd

def pairplt(data, target_col):
    """
    Create a pair plot according to color as Target column
    """
    g = sns.PairGrid(pd.concat([data.select_dtypes(exclude=['object']),data[target_col].astype(int)], axis=1), hue=target_col, diag_sharey=False)
    g = g.map_upper(plt.scatter)

    return g
    
def autolabel(rects,label,ax, label_format):
    """
    rects: [] of ax.patches
    label : []  of labels
    ax: axis to work on
    label_format: Format of label to display
    """
    # attach some text labels
    for i in xrange(len(rects)):
        x,y=rects[i].get_xy()
        height = rects[i].get_height()
        ax.text(x-4, y+.4,label_format % label[i],ha='center', va='bottom')

def plot_scatter(X,Y,trend=None):
    """
    Plots a scatter plot with a trend if passed.
    @param X: series
    @param Y: series
    @param trend: series (optional)
    """
    fig = plt.figure()
    plt.plot(X,Y,'o')
    print "Trend",trend
    if trend is not None:
        plt.plot(np.sort(X),trend,'--')
    plt.show()
    plt.clf()
    
def catg_pairplot(data,target_col, catg_col):
    """
    Create a pairplot for each category according to color as Target column 
    """
    for i in data[catg_col].unique():
        pairplt(data[data[catg_col] ==i], target_col).add_legend(title="{0}-{1}".format(catg_col,i))

def catg_countplot(data, target_col):
    """
    Creates a countplot for each category as color and target as Y-axis
    """
    plt.subplots_adjust(hspace=2, wspace=1.5)
    catg_cols = data.select_dtypes(include=['object']).columns
    fig,axs = plt.subplots((1+len(catg_cols))/2,2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    for i,col in enumerate(catg_cols):
        axis = axs[i/2,i%2]
        axis.set_title(col)
        sns.countplot(x=target_col, hue=col, data=data,ax=axis)
        

def scat_plot(data, x, y, hue):
    plt.subplots_adjust(hspace=2, wspace=1.5)
    catg_cols = data.select_dtypes(include=['object']).columns
    for i,col in enumerate(catg_cols):
        g = sns.FacetGrid(data, col=col, hue=hue)
        g.map(sns.regplot, x, y, fit_reg=False,scatter_kws={'cmap':'jet'})
        g.add_legend()
