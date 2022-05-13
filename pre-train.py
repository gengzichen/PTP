'''
Copyright (c) 2022 Zen Geng | All rights reserved | Contact gengzichenchin@gmail.com
Version: 1.1
pre-train will mainly perform the pretraining process.
This process will do the MAE work and will try to recover the masked trajectory
'''

from email.policy import strict
from pickletools import optimize
from libmain import *
from TSFMAE import *
from model import ContrastiveMAE
from torch import optim
from utils import *
from metrics import *
from statistics import mean
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR

def train(model:ContrastiveMAE, 
          optimizer:optim.Optimizer,
          device,
          dataset:Dataset,
          epochs = 20,
          valid_prop = 0.2,
          batch_size = 20,
          mask_prop=0.2):
    
    valid_size = int(len(dataset)*valid_prop)
    train_size = len(dataset) - valid_size
    
    train_set, valid_set = random_split(
        dataset=dataset,
        lengths=[train_size,valid_size]
    )
    train_loader = DataLoader(train_set,batch_size=batch_size)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.96**epoch)
    
    if os.path.exists('PretrainedMAE.pth'):
        PretrainedMAE = MaskedAutoEncoder()
        PretrainedMAE.load_state_dict(torch.load('PretrainedMAE.pth'),strict=False)
        model.MAE = PretrainedMAE
    
    for epoch in range(epochs):
        print('='*20, 'Epoch', epoch,'='*20)
        epoch_loss = []
        with tqdm(total=len(train_set), desc='Training', leave=False, ncols=100
                  ,unit='', unit_scale=True) as pbar:
            for i, (data, label) in enumerate(train_loader):
                [represent, result, rep_sample] = model(data[:,:,:,-2:].to(torch.float32).to(device), 
                                            data[:,:,:,-2:].to(torch.float32).to(device),
                                            mask_prop)
                batch_loss = ADE(result.to(device), data[:,:,0,-2:].to(device)).to(device) +\
                             0.0000001 * NCE(represent, rep_sample)
                batch_loss.backward()
                optimizer.step()
                epoch_loss.append(batch_loss.detach().numpy())
                pbar.update(batch_size)
            scheduler.step()
        print('The loss is -----', np.array(epoch_loss).mean())
        
    torch.save(model.MAE.state_dict(),'./PretrainedMAE.pth')
    
                
        
        

def preTrainArgParser():
    parser = argparse.ArgumentParser()
    # Data set arugument.
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--path', type=str, default='./datasets/eth/train')
    parser.add_argument('--max_ped', type=int, default=81)
    parser.add_argument('--sep', type=str,default='\t')
    # Model argument.
    parser.add_argument('--d_input', type=int, default=2)
    parser.add_argument('--d_spatial', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--ped_size', type=int, default=81)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_first', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Training argument.
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--mask_prop', type=float, default=0.2)
    parser.add_argument('--valid_prop', type=float, default=0.2)
    
    return parser.parse_args()

def main():
    # Parse Argument
    args = preTrainArgParser()
    # Construct Dataset
    dataset = TrajDatasets(
        dir = args.path,
        obs_len = args.obs_len,
        pred_len = args.pred_len,
        max_ped = args.max_ped
    )
    
    # Initialize device
    device = torch.device('cpu')
    
    # Instantilize model
    model = ContrastiveMAE(
        d_input=args.d_input, d_spatial=args.d_spatial,
        nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        ped_size=args.ped_size, dropout=args.dropout,
        batch_first=args.batch_first, device=device
    ).to(device)
    
    # Initialize training
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # Start training
    train(model, optimizer, device=device, dataset=dataset, epochs=args.epochs, 
          valid_prop=args.valid_prop,batch_size=args.batch_size, mask_prop=args.mask_prop)

if __name__=="__main__":
    main()