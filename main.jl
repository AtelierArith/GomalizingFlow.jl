using GomalizingFlow

hp = GomalizingFlow.load_hyperparams(
    "cfgs/example2d.toml";
    device_id=0,
    pretrained=nothing,
    result="result",
)
GomalizingFlow.train(hp)
