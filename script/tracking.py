import numpy as np
from scipy.spatial import distance
from skimage.filters import threshold_otsu

def Construct_Track(coord, max_search = 4, max_dist = 7, particle = 0,
                    nx = 0, tp=0):

    """Function use to track the blobs. The blobs are tracked bases in their
    distance between frame.

    Parameters
    ----------
    local_maxima : ndarray
        list of coordinates of blobs.
    max_search : int
        How many frame we search for a blob within max_dist.
    max_dist : int
        maximum distance to search for a blob.
    particle : int
        Which blob to track in the list of local maxima.
    nx : int
        Which sequence of the local maxima list are we working with.
    tp : Which time point are starting from.

    Returns
    -------
    Tracked particle : list
        list of coordinates.
    """

    lst_fina=[]
    lst_fina_d1 = []
    lst_fina_d2 = []

    division = None
    infected = False
    infected_d1 = False
    infected_d2 = False

    infect_time = []
    uninfect_time = []

    infect_time_d1 = []
    uninfect_time_d1 = []

    infect_time_d2 = []
    uninfect_time_d2 = []

    j = 0
    i = 0
    tp=tp
    # nx is the time point
    # particle, which particle to track, 0:3 ==> how many parameters are save with the Segmentation

    l1 = coord[nx][particle,0:5][np.newaxis, :]
    for l2 in coord[1:]:

        i += 1
        # Find all the distances between l1 and every object in the +1 frame
        dist = distance.cdist(l1[:,0:2], l2[:,0:2], 'euclidean')

        # Verify that I have less than 1 cell within max distance
        check_dist = np.where(dist[0]<max_dist)

        max_dist_m = max_dist

        if len(check_dist[0][:]) > 1:
            max_dist_m = max_dist/2

        # Verify what we have within max distance 1/2 max distance
            check_dist_b = np.where(dist[0]<max_dist_m)

            if len(check_dist[0][:]) == 0:
                max_dist_m = max_dist * 1.5

            elif len(check_dist[0][:]) > 1:
                max_dist_m =  max_dist_m/2


        #Concatenate coordinate Time frame 1 and +1 if distance is < max (make sure that all info included)
        result =  np.concatenate((l2[:,0][dist[0]< max_dist_m],
                                  l2[:,1][dist[0]< max_dist_m],
                                  l2[:,2][dist[0]< max_dist_m],
                                  l2[:,3][dist[0]< max_dist_m],
                                  l2[:,4][dist[0]< max_dist_m]), axis=0)
        #if result[4]['intensity_image'].max() == 0:
            #break

        result_tp = np.concatenate((result, (np.full(1 ,i, dtype=result.dtype))))


        #check, if during this loop I didn't find any close coordinate at T+1 for max_search above T
        if int(result.shape[0]) == 0:
            j+=1

            if j < max_search:
                continue
            else:
                break
        #if shape >2 (if I found more than 1 objecte within the max distance I stop the loop)

        elif int(result.shape[0]) > 5:
            break


        # Here I'm looking at an increase in intensity, if it happens, I will stop the loop and look for daughter cells

        elif tp > 2 and (lst_fina[tp-2][0,2])*20/100+lst_fina[tp-2][0,2] < result[2] and len(lst_fina_d1) == 0:

            # First I will track 1st daughter
            #----------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------

            division = i
            l1_d1 = result[np.newaxis, :]
            search = 0
            k = i

            for l2_d1 in coord[i+1:]:

                k += 1
                # Find all the distances between l1 and every object in the +1 frame
                dist_d1 = distance.cdist(l1_d1[:,0:2], l2_d1[:,0:2], 'euclidean')

                max_dist_d = max_dist/2

                # Verify that I have less than 1 d1 within max distance
                check_dist_d1 = np.where(dist_d1[0]<max_dist_d)

                # if 2 dist within the range use the smallest one as max distance
                if len(check_dist_d1[0][:]) == 2:
                    max_dist_d = np.ceil(dist_d1[0].min())


                if len(check_dist_d1[0][:]) > 2:
                    max_dist_d = max_dist/3

                    check_dist_d1 = np.where(dist_d1[0]<max_dist_d)

                    if len(check_dist_d1[0][:]) == 2:
                        max_dist_d = np.ceil(dist_d1[0].min())


                #Concatenate coordinate Time frame 1 and +1 if distance is < max
                result_d1 =  np.concatenate((l2_d1[:,0][dist_d1[0]<max_dist_d],
                                             l2_d1[:,1][dist_d1[0]<max_dist_d],
                                             l2_d1[:,2][dist_d1[0]<max_dist_d],
                                             l2_d1[:,3][dist_d1[0]<max_dist_d],
                                             l2_d1[:,4][dist_d1[0]<max_dist_d]), axis=0)

                #if result_d1[4]['intensity_image'].max() == 0:
                    #break

                result_tp_d1 = np.concatenate((result_d1, (np.full(1 ,k, dtype=result.dtype))))

                if int(result_d1.shape[0]) == 0:

                    # The next loop allow to increase the circle of research
                    for p in range(3):

                        if p == 0:

                            max_dist_d1 = dist_d1[0,:][dist_d1[0]<max_dist]

                            if len(max_dist_d1)>1:
                                max_dist_d1 = np.amin(max_dist_d1) + 2
                            else:
                                max_dist_d1 = max_dist

                            result_d1 =  np.concatenate((l2_d1[:,0][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,1][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,2][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,3][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,4][dist_d1[0]<max_dist_d1]), axis=0)

                            #if result_d1[4]['intensity_image'].max() == 0:
                                #break

                            result_tp_d1 = np.concatenate((result_d1, (np.full(1 ,tp, dtype=result.dtype))))

                        # if we still didn't find anything, increase the size of the circle

                        elif p == 1 and int(result_d1.shape[0]) == 0:

                            max_dist_d1 = dist_d1[0,:][dist_d1[0]<max_dist*1.5]

                            if len(max_dist_d1)>1:
                                max_dist_d1 = np.amin(max_dist_d1) + 2
                            else:
                                max_dist_d1 = max_dist*1.5

                            result_d1 =  np.concatenate((l2_d1[:,0][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,1][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,2][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,3][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,4][dist_d1[0]<max_dist_d1]), axis=0)

                            #if result_d1[4]['intensity_image'].max() == 0:
                                #break

                            result_tp_d1 = np.concatenate((result_d1, (np.full(1 ,tp, dtype=result.dtype))))

                        elif p == 2 and int(result_d1.shape[0]) == 0:

                            max_dist_d1 = dist_d1[0,:][dist_d1[0]<max_dist*2]

                            if len(max_dist_d1)>1:
                                max_dist_d1 = np.amin(max_dist_d1) + 2
                            else:
                                max_dist_d1 = max_dist*2

                            result_d1 =  np.concatenate((l2_d1[:,0][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,1][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,2][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,3][dist_d1[0]<max_dist_d1],
                                                         l2_d1[:,4][dist_d1[0]<max_dist_d1]), axis=0)
                            #if result_d1[4]['intensity_image'].max() == 0:
                                #break

                            result_tp_d1 = np.concatenate((result_d1, (np.full(1 ,tp, dtype=result.dtype))))
                # if after looking at different size circle still nothing try again at the next loop, this in case
                # the cell was not found during 1 timeframe with segmentation.

                if int(result_d1.shape[0]) == 0:
                    search+=1
                    if search < max_search:
                        continue
                    else:
                        break

                elif int(result_d1.shape[0]) > 5:
                    break


                l1_tp_d1 = np.concatenate((l1_d1, (np.full((1,1) ,k, dtype=l1_d1.dtype))), axis=1)

                lst_fina_d1.append(np.vstack((l1_tp_d1, result_tp_d1)))
                l1_d1 = result_d1[np.newaxis, :]


                # Check if daughter 1 is infected
                #--------------------------------

                #d1_inf = np.copy(result_d1[4]['intensity_image'])
                #thresh_d1 = threshold_otsu(d1_inf[d1_inf>0])
                #thresh_inf_d1 = thresh_d1 - thresh_d1*10/100


                # if there is any parasite on top of daughter 1:

                if result_d1[3]:
                    infected_d1 = True
                else:
                    infected_d1 = False

                    #for p in range(len(result_d1[3])):

                    #    sum_above_thresh_d1 = np.sum(np.logical_not(result_d1[3][p]['intensity_image'] < thresh_inf_d1))

                        #at least 35% of the surface cover by parasite is bellow threshold

                    #    if sum_above_thresh_d1 > np.size(result_d1[3][p]['intensity_image'])*50/100:
                    #        infected_d1 = True
                    #    else:
                    #        infected_d1 = False
                #else:
                #    infected_d1 = False

                if infected == True:
                    infect_time_d1.append((k, infected_d2))
                else:
                    infect_time_d1.append((k, infected_d2))


            # Now I will try to find the 2nd daughter
            #-----------------------------------------------------------------------------
            #-----------------------------------------------------------------------------
            #-----------------------------------------------------------------------------

            l1_d2 = result[np.newaxis, :]
            l = i

            search = 0

            time_d2 = 0
            for l2_d2 in coord[i+1:]:

                time_d2 += 1

                l +=1
                # Find all the distances between l1 and every object in the +1 frame

                dist_d2 = distance.cdist(l1_d2[:,0:2], l2_d2[:,0:2], 'euclidean')

                max_dist_d2 = max_dist

                check_dist_d2 = np.where(dist_d2[0]<max_dist)

                # if 2 dist within the range use the smallest one as max distance

                if len(check_dist_d2[0][:]) > 2:
                    max_dist_d2 = max_dist/2

                    check_dist_d2 = np.where(dist_d2[0]<max_dist_d2)

                    if len(check_dist_d2[0][:]) > 2:
                        max_dist_d2 = max_dist/3
                    elif len(check_dist_d2[0][:]) == 0 :
                        max_dist_d2 = max_dist/1.5

                # if 2 dist within the range use the smallest one as max distance after frame 2
                if len(check_dist_d2[0][:]) == 2 and time_d2 >= 5:
                    max_dist_d2 = np.ceil(dist_d2[0].min())


                if len(check_dist_d2[0][:]) > 2 and time_d2 >= 5:
                    max_dist_d2 = max_dist/3

                    check_dist_d2 = np.where(dist_d2[0]<max_dist_d)

                    if len(check_dist_d1[0][:]) == 2:
                        max_dist_d2 = np.ceil(dist_d2[0].min())


                #Concatenate coordinate Time frame 1 and +1 if distance is < max
                #Start with the same track as d1 with larger area
                result_d2 =  np.concatenate((l2_d2[:,0][dist_d2[0] < (max_dist_d2)],
                                             l2_d2[:,1][dist_d2[0] < (max_dist_d2)],
                                             l2_d2[:,2][dist_d2[0] < (max_dist_d2)],
                                             l2_d2[:,3][dist_d2[0] < (max_dist_d2)],
                                             l2_d2[:,4][dist_d2[0] < (max_dist_d2)]), axis=0)

                #if result_d2[4]['intensity_image'].max() == 0:
                    #break

                result_tp_d2 = np.concatenate((result_d2, (np.full(1 ,l, dtype=result.dtype))))


                #if in this larger area there is a 2nd cell, I will try to find it and track it as d2
                if int(result_d2.shape[0]) == 10 and time_d2 < 5:

                    # If there is 2 in the bigger area, the smalest one should be d1 then the biggest one is d2
                    # only in the first 2 frame after lookinf for d2

                    max_dist_d2_b = dist_d2[0,:][dist_d2[0]<max_dist_d2]

                    max_dist_min = np.ceil(np.amin(max_dist_d2_b))

                    result_d2 =  np.concatenate((l2_d2[:,0][(max_dist_min < dist_d2[0]) & (dist_d2[0] < (max_dist_d2))],
                                                 l2_d2[:,1][(max_dist_min < dist_d2[0]) & (dist_d2[0] < (max_dist_d2))],
                                                 l2_d2[:,2][(max_dist_min < dist_d2[0]) & (dist_d2[0] < (max_dist_d2))],
                                                 l2_d2[:,3][(max_dist_min < dist_d2[0]) & (dist_d2[0] < (max_dist_d2))],
                                                 l2_d2[:,4][(max_dist_min < dist_d2[0]) & (dist_d2[0] < (max_dist_d2))]), axis=0)

                    #if result_d2[4]['intensity_image'].max() == 0:
                        #break

                    result_tp_d2 = np.concatenate((result_d2, (np.full(1 ,l, dtype=result.dtype))))
                # if I didn't find anything in the first round, I will try to increase the area and see if I find something

                if int(result_d2.shape[0]) == 0:

                # The next loop allow to increase the circle of research
                    for p in range(3):

                        if p == 0:


                            max_dist_d2 = dist_d2[0,:][dist_d2[0]<max_dist*1.5]


                            if max_dist_d2.any():

                                max_dist_min = np.ceil(np.amin(max_dist_d2))

                                if len(max_dist_d2)>1:
                                    max_dist_d2 = np.ceil(np.amin(max_dist_d2))
                                else:
                                    max_dist_d2 = max_dist



                                result_d2 =  np.concatenate((l2_d2[:,0][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*1.5))],
                                                             l2_d2[:,1][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*1.5))],
                                                             l2_d2[:,2][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*1.5))],
                                                             l2_d2[:,3][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*1.5))],
                                                             l2_d2[:,4][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*1.5))]), axis=0)

                                #if result_d2[4]['intensity_image'].max() == 0:
                                    #break

                                result_tp_d2 = np.concatenate((result_d2, (np.full(1 ,tp, dtype=result.dtype))))
                            else:
                                pass

                        # if we still didn't find anything, increase the size of the circle

                        elif p == 1 and int(result_d2.shape[0]) == 0:


                            max_dist_d2 = dist_d2[0,:][dist_d2[0]<max_dist*2]

                            if max_dist_d2.any():
                                max_dist_min = np.ceil(np.amin(max_dist_d2))

                                if len(max_dist_d2)>1:
                                    max_dist_d2 = np.ceil(np.amin(max_dist_d2))
                                else:
                                    max_dist_d2 = max_dist*1.5

                                result_d2 =  np.concatenate((l2_d2[:,0][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2))],
                                                             l2_d2[:,1][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2))],
                                                             l2_d2[:,2][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2))],
                                                             l2_d2[:,3][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2))],
                                                             l2_d2[:,4][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2))]), axis=0)

                                #if result_d2[4]['intensity_image'].max() == 0:
                                    #break
                                result_tp_d2 = np.concatenate((result_d2, (np.full(1 ,tp, dtype=result.dtype))))
                            else:
                                pass

                        elif p == 2 and int(result_d1.shape[0]) == 0:

                            max_dist_d2 = dist_d2[0,:][dist_d2[0]<max_dist*2.5]
                            if max_dist_d2.any():

                                max_dist_min = np.ceil(np.amin(max_dist_d2))

                                if len(max_dist_d2)>1:
                                    max_dist_d2 = np.ceil(np.amin(max_dist_d2))
                                else:
                                    max_dist_d2 = max_dist*2

                                result_d2 =  np.concatenate((l2_d2[:,0][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2.5))],
                                                             l2_d2[:,1][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2.5))],
                                                             l2_d2[:,2][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2.5))],
                                                             l2_d2[:,3][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2.5))],
                                                             l2_d2[:,4][(max_dist_d2 < dist_d2[0]) & (dist_d2[0] < (max_dist*2.5))]), axis=0)

                                #if result_d2[4]['intensity_image'].max() == 0:
                                    #break
                                result_tp_d2 = np.concatenate((result_d2, (np.full(1 ,tp, dtype=result.dtype))))
                            else:
                                pass

                if int(result_d2.shape[0]) == 0:
                    search+=1


                    if search < max_search:
                        continue
                    else:
                        break

                elif int(result_d2.shape[0]) > 5:
                    break

                l1_tp_d2 = np.concatenate((l1_d2, (np.full((1,1) ,l, dtype=l1_d2.dtype))), axis=1)
                lst_fina_d2.append(np.vstack((l1_tp_d2, result_tp_d2)))
                l1_d2 = result_d2[np.newaxis, :]

                # Check if daughter 2 is infected
                #--------------------------------

                #d2_inf = np.copy(result_d2[4]['intensity_image'])
                #thresh_d2 = threshold_otsu(d2_inf[d2_inf>0])
                #thresh_inf_d2 = thresh_d2 - thresh_d2*10/100




                # if there is any parasite on top of daughter 2:

                if result_d2[3]:
                    infected_d2 = True
                    #for m in range(len(result_d2[3])):

                    #    sum_above_thresh_d2 = np.sum(np.logical_not(result_d2[3][m]['intensity_image'] < thresh_inf_d2))

                        #at least 35% of the surface cover by parasite is bellow threshold

                    #    if sum_above_thresh_d2 > np.size(result_d2[3][m]['intensity_image'])*50/100:
                    #        infected_d2 = True
                    #    else:
                    #        infected_d2 = False
                else:
                    infected_d2 = False

                if infected == True:
                    infect_time_d2.append((l, infected_d2))
                else:
                    infect_time_d2.append((l, infected_d2))



        elif tp == division:
            break


        l1_tp = np.concatenate((l1, (np.full((1,1) ,i, dtype=l1.dtype))), axis=1)

        lst_fina.append(np.vstack((l1_tp, result_tp)))
        l1 = result[np.newaxis, :]

        # Check if mother cells are infected
        #--------------------------------

        #mom_inf = np.copy(result[4]['intensity_image'])
        #show_img(mom_inf)
        #thresh = threshold_otsu(mom_inf[mom_inf>0])
        #thresh_inf = thresh - thresh*10/100


         # if there is any parasite on top of the mother:

        if result[3]:
            infected = True

            #for n in range(len(result[3])):
            #    sum_above_thresh = np.sum(np.logical_not(result[3][n]['intensity_image'] < thresh_inf))
                #at least 35% of the surface cover by parasite is bellow threshold
            #    if sum_above_thresh > np.size(result[3][n]['intensity_image'])*50/100:
            #        infected = True
            #    else:
            #        infected = False
        else:
            infected = False

        if infected == True:
            infect_time.append((tp, infected))

        else:
            uninfect_time.append((tp, infected))

        tp+=1
    arr = np.asarray(lst_fina)
    arr_d1 = np.asarray(lst_fina_d1)
    arr_d2 = np.asarray(lst_fina_d2)

    #mother
    infect_time.extend(uninfect_time)
    infect_time.sort()

    timeline_infect = np.asarray(infect_time)[:,1].astype(bool)
    mom = arr[:,0,:]
    mom = np.concatenate((mom, timeline_infect[:, np.newaxis]), axis=1)

    #daughter 1
    if int(arr_d1.shape[0]) == 0:
        daughter_1 = arr_d1
    else:

        infect_time_d1.extend(uninfect_time_d1)
        infect_time_d1.sort()
        timeline_infect_d1 = np.asarray(infect_time_d1)[:,1].astype(bool)

        daughter_1 = arr_d1[:,0,:]
        daughter_1 = np.concatenate((daughter_1, timeline_infect_d1[:, np.newaxis]), axis=1)

    #daughter 2
    if int(arr_d2.shape[0]) == 0:
        daughter_2 = arr_d2
    else:
        infect_time_d2.extend(uninfect_time_d2)
        infect_time_d2.sort()
        timeline_infect_d2 = np.asarray(infect_time_d2)[:,1].astype(bool)

        daughter_2 = arr_d2[:,0,:]
        daughter_2 = np.concatenate((daughter_2, timeline_infect_d2[:, np.newaxis]), axis=1)


    return(mom, daughter_1, daughter_2)

def _clean_d2(liste_a):
    for i in range(len(liste_a)):
        for j in range(len(liste_a)):
            #print(j)
            try:
                ar_daughter2 = liste_a[j][2][:,0:2]
                ar_mom = liste_a[i][0][:,0:2]
                intersect = np.in1d(ar_daughter2, ar_mom)
                intersect_b = intersect.reshape(int(len(intersect)/2), 2)

                liste_a[j][2][:,0:2][intersect_b]=0
            except IndexError:
                pass
    for k in range(len(liste_a)):
        try:
            ar = (liste_a[k][2][:,0:2])
            liste_a[k][2][:,0:2][np.where(ar==0)] = np.nan
        except IndexError:
            pass
    return liste_a

def track_multi(img, result_seg):
    liste_a=[]

    nt, nx, ny = img.shape

    for x in range(len(result_seg[0])):

        try:
            Mother, daughter_1, daughter_2 = Construct_Track(result_seg, max_search = 5, max_dist = 100,
                                             particle = x, nx = 0, tp=0)
            familly = [Mother, daughter_1, daughter_2]

        except IndexError:
            pass

        liste_a.append(familly)

    return _clean_d2(liste_a)
