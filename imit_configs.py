from easydict import EasyDict as edict

IMIT_CONFIGS = {
    "sub7_walk1":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/sub7_walk1/1st/logs/",
        "log_interval":10,
        "filename":"./motionData/humanoid_CMU/subject7_walk1.amc",
        "model_dir":"./expData/sub7_walk1/1st/policies/",
        "max_num_frames":316,
        "hidden_size": [512, 256],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":2*1e-5,
        "critic_lr":1e-5,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":200,
        "batch_size":10000,
        "total_sample_size":100000


    })
}
