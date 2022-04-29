using LFT

hp = LFT.load_hyperparams("cfgs/example2d.toml"; device_id=0, pretrained=nothing, result="result")
LFT.train(hp)
