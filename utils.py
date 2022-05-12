'''
Copyright (c) 2021 Zen Geng | All rights reserved | Contact gengzichenchin@gmail.com
Version: 1.0
utils contain mainly 1 interface:
1. TrajDatasets
'''

from os import times

from libmain import *
from torch.utils.data import Dataset


import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def readDataFile(filename, sep = '\t'):
    '''Read Data file in form of list of list from given file'''
    result = []
    with open(filename, 'r') as FILE:
        for line in FILE:
            line = line.strip().split(sep)
            line = list(map(float, line))
            result.append(line)
        return np.array(result)

def vecPedAround(pedAround, frames, maxPed=80):
    '''Args: pedAround should in the form of np.array 
             [seqLen, , 4]
             frames is the frame number of the given ped.                                                                                 
             
    '''
    pedToken = np.zeros((len(frames), maxPed, 4))
    for timestamp, frame in enumerate(frames):
        curPedAround = pedAround[pedAround[:,0] == frame, :] \
        if frame in pedAround[:,0] else None
        if curPedAround is not None: 
            assert curPedAround.shape[0] <= maxPed
            pedToken[int(timestamp),0:curPedAround.shape[0],:] = curPedAround[:,]
    return pedToken



class TrajDatasets(Dataset):
    '''Datasets class for Trajectory data from given path'''
    def __init__(self, dir, obs_len = 10, pred_len = 10, max_ped = 81,
                 sep = '\t', norm = True, rel = True, image = False) -> None:
        '''
        Args: 
        * dir: directory containing data files of coordinates
        The data file should have the following structure:
        [frameID, pedestrianID, x, y] for each line
        * obs_len: Observed sequence length, 10 by default
        * pred_len: Prediction sequence length, 10 by default
        * sep: separator.
        * norm: whether coordinates are [0,1] normed.
        * rel: whether surroundings use relative coordinate.
        * image: whether use image.
        '''
        super().__init__()
        self.dir = dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.sep = sep
        self.norm = norm
        self.image = image
        self.maxPed = max_ped
        self.seq_len = obs_len + pred_len
        self.seq_set = [] # Main Result for batches
        files = [os.path.join(self.dir, path) for path in os.listdir(self.dir)]
        for filename in files:
            data = readDataFile(filename=filename, sep=self.sep)
            frames = np.unique(data[:, 0]).tolist()
            frameData = []
            for frame in frames:
                frameData.append(data[frame == data[:, 0], :])
                # frameData in the form of list of array (,4), group by FrameID
                # at dim 0.
            seqNum = len(frames) - self.seq_len + 1
            for frameID in range(0, len(frames)):
                seqData = np.concatenate(frameData[frameID:frameID+self.seq_len])
                seqPed = np.unique(seqData[:,1])
                self.maxPed = max(len(seqPed), self.maxPed)
                
                for ped, pedID in enumerate(seqPed):
                    targetSeq = seqData[seqData[:,1] == pedID, :]
                    if len(targetSeq) != self.seq_len: continue # Not consider as target
                    targetFrames = targetSeq[:, 0]
                    pedAround = seqData[seqData[:,1]!=pedID, :]
                    pedAroundToken = vecPedAround(pedAround, targetFrames, 80)
                    pedTrajGraph = np.concatenate([targetSeq.reshape((20,1,4)),pedAroundToken],axis=1)
                    self.seq_set.append(pedTrajGraph)
                    
    def __getitem__(self, index):
        return self.seq_set[index][0:self.obs_len], self.seq_set[index][self.obs_len:]
    def __len__(self):
        return len(self.seq_set)



def plotSeqDiagram(graph):
    '''
    The graph should be a tensor in the form of [seqLen, maxPed+1, 4]
    '''
    #numValidPed = int(graph[0,graph[0]!=torch.zeros(4)].shape[0]/2)
    for frame in range(0, graph.shape[0]):
        for ped in range(0, graph.shape[1]):
            if ped == 0:
                print(graph[frame][ped])
                plt.plot(graph[frame][ped][-2].numpy(), 
                        graph[frame][ped][-1].numpy(), "ob",markersize = 3+0.5*ped)
            else:
                if torch.sum(graph[frame,ped,:])!=0:
                    plt.plot(graph[frame][ped][-2].numpy(), 
                            graph[frame][ped][-1].numpy(), "or",markersize = 1+0.5*ped)
    plt.show()
