import os, importlib
import numpy as np
import copy

def read_ait_h_instance(file_to_read, folder='ait _haddadene_et_al_2016'):
    ## This bit does not need changing:
    inst = importlib.import_module(folder + '.' + file_to_read)
    inst.od = depl_to_od(inst)
    inst.CoutPref = np.array(inst.CoutPref)
    return inst

def depl_to_od(ldi):
    d = []
    normalRow = ldi.NbClients - 1
    extRow = ldi.NbClients
    finalRow = ldi.depl[-1*extRow:]
    # print(ldi.depl)
    # print(finalRow)
    for ii in range(ldi.NbClients + 1):
        if ii == 0:
            map_i = ldi.NbClients
            rowSize = extRow
        else:
            map_i = ii - 1
            rowSize = normalRow
        stLim = rowSize*map_i
        endLim = rowSize*(map_i + 1)
        dlRow = copy.copy(ldi.depl[stLim:endLim])

        if ii != 0:
            # print('Inserting ' + str(map_i) + ' / ' + str(len(finalRow)))
            dlRow.insert(0,finalRow[map_i])
            dlRow.insert(map_i + 1, 0)
        else:
            dlRow.insert(0, 0)

        d.append(copy.copy(dlRow))
    return np.array(d)