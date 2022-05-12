from libmain import *

def ADE(x, y):
    '''Args: x should be tensor in the form of [seqlen,pedNum, 2]'''
    return torch.mean(torch.sqrt(torch.sum(torch.pow((x - y),2), dim=-1)))
    

def FDE(x, y):
    '''Args: The same as ADE()'''
    return torch.sum(torch.sqrt(torch.sum(torch.pow((x - y),2), dim=-1)),dim=0)


def NCE(rep, sample, temp=1000):
    ''' Args:
        * rep: The representation of anchor in shape [N, S, Rep].
        * sample: in the form of [N*(1+NegNumber), S, Rep].
        * temp: temperature variable to control the similarity speed.
    '''
    N = rep.shape[0]
    NegNumber = int((sample.shape[0]-N)/N)
    pos_similarity = similarity(rep, sample[0:N,:,:])
    neg_similarity = torch.zeros((N,NegNumber, sample.shape[1]))
    for i in range(NegNumber):
        neg_similarity[:,i,:] = similarity(rep, sample[(i+1)*N:(i+1)*N+N])
    pos_similarity = torch.exp(pos_similarity/temp)
    neg_similarity = torch.exp(neg_similarity/temp)
    neg_similarity = torch.sum(neg_similarity, dim=1)
    result = torch.mean(pos_similarity/neg_similarity)
    return result
    
    

def similarity(x, y):
    '''Return cosine similarity x, y in [N, S, E]'''
    return torch.einsum('NSE,NSE->NS',x, y)
    