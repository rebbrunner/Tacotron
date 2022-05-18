import torch
import soundfile as sf
from univoc import Vocoder
# from tacotron import Tacotron
from MyTaco import Tacotron
from text_tools import load_cmudict, text_to_id

vocoder = Vocoder.from_pretrained(
    "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt"
).cuda()

tacotron = Tacotron.from_pretrained(
    "C:/Users/Rebs/Documents/Dev/data-science-env/data/TTS/tacotron-ljspeech-yspjx3.pt"
).cuda()

cmudict = load_cmudict()
cmudict["PYTORCH"] = "P AY1 T AO2 R CH"
cmudict['REBECCA'] = 'R EH1 B EH0 AH0 K AA1'

text = "Hello, Rebecca.  How are you today?"

x = torch.LongTensor(text_to_id(text, cmudict)).unsqueeze(0).cuda()
with torch.no_grad():
    mel, alpha = tacotron.generate(x)
    wav, sr = vocoder.generate(mel.transpose(1, 2))

sf.write('./test.wav', wav, sr)
