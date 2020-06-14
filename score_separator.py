import pandas as pd 

path = "Alexnet_21-03-20:38_score"
csv = pd.read_csv('outputs/'+path +'.csv')

newset = {"Real":[],"Predicted":[],"Prob":[]}
for index, val in csv.iterrows():
    real = val["Real"].split("(")[1].split(",")[0]
    predict = val["Predicted"].split("(")[1].split(",")[0]
    prob = val["Prob"].split("(")[1].split(",")[0]

    newset["Real"].append(real)
    newset["Predicted"].append(predict)
    newset["Prob"].append(prob)

result_df = pd.DataFrame(
    newset, columns=['Real', 'Predicted','Prob'])
result_df.to_csv('outputs/'+path +'.csv', index=False)
