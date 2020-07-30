import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from scripts.model import *
from scripts.dataset import *
from scripts.parse_config import *
from scripts.utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4 , help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="data/PAE.data", help="path to data config file")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    opt = parser.parse_args()
    
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    EPOCHS = opt.epochs
    BATCH_SIZE = opt.batch_size
    N_CPU = opt.n_cpu

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
  
    model = PAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))

    dataset = ListDataset(train_path, img_size = 640)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    #loss_values = []
    for epoch in range(opt.epochs):
        model.train()
        for batch_idx, (_, imgs, targets) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_idx
            
            imgs = Variable(imgs.to(DEVICE))
            targets = Variable(targets.to(DEVICE), requires_grad=False)
            
            output, loss = model(imgs, targets)
            loss.backward()
            
            if batches_done % opt.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            #loss_values.append(loss.item())
            #plt.plot(loss_values)
            #plt.pause(0.05)
            
            print('[{}][{}] Test Loss : {:.4f}'.format(epoch, batch_idx, loss))
        
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/pae_ckpt_%d.pth" % epoch)
        
        #plt.show()

            
