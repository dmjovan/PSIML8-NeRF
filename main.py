import yaml
import argparse
import time

from NeRF.datasets.replica import replica_dataset
from NeRF.training import trainer

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file",
                        type=str, 
                        default=r"C:\Users\psiml8\Desktop\Project\NeRF\configs\room0_config.yaml", 
                        help="config file name")

    parser.add_argument("--dataset_type", 
                        type=str, 
                        default="replica", 
                        choices= ["replica", "lego"], 
                        help="the dataset to be used")

    parser.add_argument("--gpu", 
                        type=str, 
                        default="", 
                        help="GPU IDs")

    args = parser.parse_args()

    # read YAML file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # updating config with GPUs
    if len(args.gpu) > 0:
        config["experiment"]["gpu"] = args.gpu

    trainer.select_gpus(config["experiment"]["gpu"])
    config["experiment"].update(vars(args))
    
    # Creating NeRFTrainer
    nerf_trainer = trainer.NeRFTrainer(config)

    if args.dataset_type == "replica":
        print("####################################################################################")
        print("------------------------------- Using Replica Dataset ------------------------------")

        total_num = 900 # total number of images in dataset
        step = 5
        train_ids = list(range(0, total_num, step)) # training images - every 5th image
        test_ids = [x + (step//2) for x in train_ids] #  test images - images between two training images

        # FIXME: actually using only 180 images for training -> should we increase the number of images 

        config["experiment"]["train_ids"] = train_ids
        config["experiment"]["test_ids"] = test_ids

        replica_data_loader = replica_dataset.ReplicaDataset(data_dir=config["experiment"]["dataset_dir"],
                                                             train_ids=train_ids, 
                                                             test_ids=test_ids,
                                                             img_h=config["experiment"]["height"],
                                                             img_w=config["experiment"]["width"])

        nerf_trainer.set_params_replica()
        nerf_trainer.prepare_data_replica(replica_data_loader)
    
        # create nerf model, initialize optimizer
        nerf_trainer.create_nerf()

        # create rays in world coordinates
        nerf_trainer.init_rays()

        N_iters = int(float(config["train"]["N_iters"])) + 1

        print("#################################################################################")
        print("-------------------------- Begining og trainining loop -------------------------")

        for i in range(0, N_iters):

            step_start_time = time.time()
            nerf_trainer.step(i)
            step_end_time = time.time()

            step_duration = step_end_time - step_start_time
            print("Step duration is :", step_duration)
