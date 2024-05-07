import torch
import yaml
from src.models.whisper_lcnn import WhisperLCNN
from whisper_preprocessor import output 
from detectron2.config import get_cfg


with open("D:\\vit btech final year 2023\\Capstone\\Pretrained models\\all_models\\whisper_lcnn\\config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

print(model_config)


"""
parameters= model_config['model']['parameters']
#model_config['model']['parameters']['device'] = "cpu"
#print(parameters)
model = WhisperLCNN(
    input_channels=1,
    freeze_encoder=True,
    device='cpu',  # Add device argument here
)
print(model)
cfg = get_cfg()
cfg.merge_from_other_cfg(model_config)
"""



model_config.load_state_dict(torch.load(("D:\\vit btech final year 2023\\Capstone\\Pretrained models\\all_models\\whisper_lcnn\\weights.pth"),map_location=torch.device('cpu')))
print(model_config)


audio_features_tensor = output 


with torch.no_grad():
    model.eval()  
    logits = model(audio_features_tensor)


probabilities = torch.sigmoid(logits)  
predicted_class = 1 if probabilities > 0.5 else 0  

if predicted_class == 1:
    print("The audio is classified as spoof.")
else:
    print("The audio is classified as bonafide.")
