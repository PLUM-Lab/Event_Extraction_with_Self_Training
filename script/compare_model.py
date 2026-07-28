import json

score_path = "YOUR_SCORE_RESULT_PATH"
ie_path = "YOUR_IE_RESULT_PATH"


with open(score_path,'r') as fin:
    score_result = json.load(fin)
with open(ie_path,'r') as fin:
    ie_result = json.load(fin)
total = 0
for s_k, s_v in score_result.items():
    if s_k == 'O':
        continue
    i_v = ie_result[s_k]
    print("label: {}, F1: {}, Tp:{}, FP {}, FN {}, total number: {}".format(s_k, (s_v["F1"]-i_v["F1"])*100,  s_v["tp"]-i_v["tp"], s_v["fp"]-i_v["fp"],s_v["fn"]-i_v["fn"] ,s_v['tp']+s_v['fn']))
    total+=(s_v["F1"]-i_v["F1"])*(s_v['tp']+s_v['fn'])
print(total)
