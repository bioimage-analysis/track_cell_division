import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np

def browse_track_multi(img, liste_a, segmentation):
    nt, ny, nx = img.shape
    def plot_track(i, save=False, name_img = "file.png"):
        fig, axes = plt.subplots(1,1, figsize=(8, 8))
        axes.imshow(img[i], cmap="gray",interpolation='nearest')
        #axes.contour(segmentation[i], [0.5], linewidths=1.2, colors='y')

        color=iter(plt.cm.jet(np.linspace(0,1,len(liste_a))))

        for n in range(len(liste_a)):
            c=next(color)
            # plot the mother track while i increase
            axes.plot(liste_a[n][0][0:i+1,1], liste_a[n][0][0:i+1,0], linewidth=3, c=c)
            #axes.text(liste_a[n][0][0,1]+50, liste_a[n][0][0,0]+5, "cell_{}".format(n+1), fontsize = 14, color = c)

            # plot the daughter 1 track while i increase, if there is a daughter1 and if we have reach the end mom.

            if len(liste_a[n][1]) > 0 and i > len(liste_a[n][0]):

                axes.plot(liste_a[n][1][0:i-len(liste_a[n][0])+1,1],
                          liste_a[n][1][0:i-len(liste_a[n][0])+1,0],
                          '--', linewidth=3, c='m')
                #axes.text(liste_a[n][1][i-len(liste_a[n][0]),1]+50,
                #          liste_a[n][1][i-len(liste_a[n][0]),0]+5,
                #          'd1', fontsize = 12, color = 'g')
            else:
                pass
            # plot the daughter 2 track while i increase, if there is a daughter2 and if we have reach the end mom.

            if len(liste_a[n][2]) > 0 and i > len(liste_a[n][0]):

                axes.plot(liste_a[n][2][0:i-len(liste_a[n][0])+1,1],
                          liste_a[n][2][0:i-len(liste_a[n][0])+1,0],
                          '--', linewidth=3, c='w')
               # axes.text(liste_a[n][2][i-len(liste_a[n][0]),1]+50,
               #           liste_a[n][2][i-len(liste_a[n][0]),0]+5,
               #           'd2', fontsize = 12, color = 'b')
            else:
                pass

        axes.axis("off")
        axes.autoscale_view('tight')
        plt.show()
        if save == True:
            fig.savefig(name_img, bbox_inches='tight')
    interact(plot_track, i=(0,nt-1), save=False,name_img = "file.png")

def browse_track(img, mom, d1, d2, result,segmentation):
    nt, ny, nx = img.shape
    def plot_track(i):
        fig, axes = plt.subplots(1,1, figsize=(12, 12))
        axes.imshow(img[i], cmap="gray",interpolation='nearest')
        axes.contour(segmentation[i], [0.5], linewidths=1.2, colors='y')

        for coor in result:

            axes.scatter(coor[:,1], coor[:,0], s=4, c='g')


        axes.plot(mom[0:i+1,1], mom[0:i+1,0], linewidth=1, c='r')
        if i < len(mom):
            axes.text(mom[i,1]+50, mom[i,0]+5, "mom", fontsize = 14, color = 'r')

        if i > len(mom):

            axes.plot(d1[0:i-len(mom)+1,1], d1[0:i-len(mom)+1,0], linewidth=2, c='b')
            axes.text(d1[i-len(mom),1]+50, d1[i-len(mom),0]+5, "d1", fontsize = 12, color = 'b')

            axes.plot(d2[0:i-len(mom)+1,1], d2[0:i-len(mom)+1,0], linewidth=2, c='g')
            #axes.text(d2[i-len(mom),1]+50, d2[i-len(mom),0]+5, "d2", fontsize = 12, color = 'g')



        axes.axis("off")
        axes.autoscale_view('tight')
        plt.show()

    interact(plot_track, i=(0,nt-1))
