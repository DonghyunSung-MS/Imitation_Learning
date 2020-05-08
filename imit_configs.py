from easydict import EasyDict as edict

IMIT_CONFIGS = {
    "sub8_walk1":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/sub8_walk1/2st/logs/",
        "log_interval":1,
        "save_interval":30,
        "filename":"./motionData/humanoid_CMU/sub8_walk1.amc",
        "model_dir":"./expData/sub8_walk1/2st/policies/",
        "max_num_frames":277,
        "hidden_size": [512, 256],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":2*1e-5,
        "critic_lr":1e-5,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":10,
        "batch_size":100,
        "total_sample_size":1000


    })
}
