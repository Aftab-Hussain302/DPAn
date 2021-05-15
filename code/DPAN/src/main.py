import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
print('hello world')
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)

        params = list(model.parameters())
        params_sum = 0
        for param in params:          
            temp = 1
            for dim in param.size():                 
                temp *= dim             
            params_sum += temp
        print("number of params:%d" % params_sum)
        
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

