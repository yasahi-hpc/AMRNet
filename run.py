from helpers.parser import parse
from model.trainers import get_trainer
import time

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name
    n_epochs = args.n_epochs
    args.data_dir = '/home/1/17IKA143/jhpcn2021/work/Deeplearning/FlowCNN/SteadyFlow/AMR_Net/dataset/datasets/steady_flow_Re20_v8'
    trainer = get_trainer(model_name)(**vars(args))

    trainer.initialize()

    total_start = time.time()
    for epoch in range(n_epochs):
        trainer.step(epoch)

    seconds = time.time() - total_start
    trainer.finalize(seconds)
