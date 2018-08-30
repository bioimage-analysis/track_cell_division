import pandas as pd
import numpy as np

def create_pandas_track(liste_a, path='', save=False):

    name_cell = []
    families = []
    for x in range(len(liste_a)):

        name_cell.append("cell_{}".format(x+1))
        mom = pd.DataFrame(liste_a[x][0][:,0:2], columns = ['x', 'y'])
        mom['Numb of para'] = pd.Series([len(i) for i in liste_a[x][0][:,3]], index=mom.index)
        mom['time point'] = pd.Series(liste_a[x][0][:,5], index=mom.index)
        mom['infected'] = pd.Series(liste_a[x][0][:,6], index=mom.index)


        if liste_a[x][1].any():
            daughter_1 = pd.DataFrame(liste_a[x][1][:,0:2], columns = ['x', 'y'])
            daughter_1['Numb of para'] = pd.Series([len(i) for i in liste_a[x][1][:,3]], index=daughter_1.index)
            daughter_1['time point'] = pd.Series(liste_a[x][1][:,5], index=daughter_1.index)
            daughter_1['infected'] = pd.Series(liste_a[x][1][:,6], index=daughter_1.index)

        else:
            daughter_1 = pd.DataFrame(  np.full((1,5), np.nan), columns = ['x', 'y', 'Numb of para',
                                                                           'time point', 'infected'])

        if liste_a[x][2].any():
            daughter_2 = pd.DataFrame(liste_a[x][2][:,0:2], columns = ['x', 'y'])
            daughter_2['Numb of para'] = pd.Series([len(i) for i in liste_a[x][2][:,3]], index=daughter_2.index)
            daughter_2['time point'] = pd.Series(liste_a[x][2][:,5], index=daughter_2.index)
            daughter_2['infected'] = pd.Series(liste_a[x][2][:,6], index=daughter_2.index)

        else:
            daughter_2 = pd.DataFrame(  np.full((1,5), np.nan), columns = ['x', 'y', 'Numb of para',
                                                                           'time point', 'infected'])


        result_family = pd.concat([mom, daughter_1, daughter_2], keys = ['mother', 'daughter 1', 'daughter 2'])
        families.append(result_family)

        dataframe = pd.concat(families, keys = name_cell)

    if save == True:
        import re
        import time
        import os

        counter = time.strftime("_%Y%m%d_%H%M")
        path, file_name = os.path.split(path)
        filename, extension = os.path.splitext(file_name)
        filename = re.match(r'.+\d+', filename)
        filename = filename.group()
        if not os.path.exists(filename):
            os.makedirs(filename)
        file = os.path.join(filename + str(counter) + ".xlsx")
        #writer = pd.ExcelWriter(file, engine='xlsxwriter')
        dataframe.to_excel("{}/{}".format(filename, file), sheet_name='Sheet1')
        #writer.save()


    return(dataframe)
