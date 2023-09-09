from utils.gpt import embedding
import numpy
# text1 = "\"\""

text1 = "\"签了劳动合同后劳动者可以提前辞职。劳动者可以提前三十天向用人单位发出书面通知，要求辞职；或者在试用期提前三天发出通知，到期后劳动合同就解除，双方可办理解约手续。\""
text2 = "\"依据《中华人民共和国劳动合同法》第三十七条劳动者提前三十日以书面形式通知用人单位，可以解除劳动合同。劳动者在试用期内提前三日通知用人单位，可以解除劳动合同。\""
text1 = embedding(text1)
text2 = embedding(text2)
print(numpy.dot(text1, text2))
