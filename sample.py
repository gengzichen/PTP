'''
Copyright (c) 2022 Zen Geng | All rights reserved | Contact gengzichenchin@gmail.com
Version: 1.1
sample contains mainly 1 interface:
1. SampleGenerator
'''

from libmain import *

class SampleGenerator(object):
    def __init__(self, noise = 0.05, comfortRadius = 0.5,
                 surNum = 8) -> None:
        '''
        * noise is the constant to avoid of overfitting, picked as 0.05, 
        * comfortRadius is 0.5 meter by default,
        * surNum is 8 for 8 directions.
        '''
        self.comfortRadius = comfortRadius
        self.noise = noise
        self.surNum = surNum
        
    def _surZone(self) -> torch.tensor:
        zone = torch.zeros((self.surNum,2))
        for i in range(self.surNum):
            zone[i, 0] = self.comfortRadius*np.cos(2*math.pi*i/self.surNum) + \
                                   random.gauss(0,self.noise)
            zone[i, 1] = self.comfortRadius*np.sin(2*math.pi*i/self.surNum) +  \
                                   random.gauss(0,self.noise)
        return zone
                              
    def _sampleNeg(self, graph:torch.tensor):
        negSample = torch.zeros((graph.shape[0], 
                                 graph.shape[1]*self.surNum, 
                                 graph.shape[2]))
        for frame in range(graph.shape[0]):
            for i in range(graph.shape[1]):
                if torch.sum(torch.abs(graph[frame, i, :]))!=0: # No sampling for null ped
                    negSample[frame,self.surNum*i:self.surNum*i+self.surNum,:] = \
                        torch.add(graph[frame,i:i+1,:],self._surZone())

        # for frame in range(graph.shape[0]):
        #     for i in range(graph.shape[1]):
        #         if torch.sum(torch.abs(graph[frame, i, :]))!=0:
        #             plt.plot(graph[frame, i, 0], graph[frame,i,1], "or")
        #             for j in range(self.surNum):
        #                 plt.plot(negSample[frame, i*8+j, 0], negSample[frame,i*8+j,1], "oy")
        return negSample
            
    def _samplePos(self, graph:torch.tensor):
        posSample = np.zeros([graph.shape[0], 1, 2])
        for frame in range(graph.shape[0]):
            posSample[frame,0,:] = graph[frame,0:1,:] + torch.tensor([random.gauss(0,self.noise),
                                            random.gauss(0,self.noise)])
        return posSample
        
    def generate(self, graph):
        '''
        graph should be in the form of [seqLen, 1 + surPedNum, 2]
        '''
        negSample = self._sampleNeg(graph[:,1:,:])
        posSample = self._samplePos(graph[:,0:1,:])
        return (negSample, posSample)
    
    def _plotSampleGraph(self, graph):
        negSample, posSample = self.generate(graph)
        graph = graph.numpy()
        for frame in range(graph.shape[0]):
            if graph[frame,0,:].sum() != 0: # No plot for null ped
                plt.plot(graph[frame, 0,0], graph[frame, 0,1], "ob", markersize = 4)
            if posSample[frame, 0,:].sum() != 0:
                plt.plot(posSample[frame, 0,0], posSample[frame, 0,1], "og",markersize = 4)
            for pedIdx in range(graph.shape[1]-1): 
                if graph[frame, pedIdx+1, :].sum()!=0: # No plot for null ped
                    plt.plot(graph[frame, pedIdx+1, 0],graph[frame, pedIdx+1, 1],
                            "oy", markersize = 3)
                    for sur in range(self.surNum):
                        plt.plot(negSample[frame, pedIdx*8+sur, 0], negSample[frame, pedIdx*8+sur, 1],
                                "or", markersize = 3)
        plt.show()