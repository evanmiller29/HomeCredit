import pandas as pd
from datetime import datetime

from os.path import join

time_now = datetime.now().strftime("%d_%m_%y_%H_%m")
output_path = 'F:/Nerdy Stuff/Kaggle submissions/DonorChoose/'

test = pd.read_csv('F:/Nerdy Stuff/Kaggle/Home credit/data/application_test.csv')
preds = pd.read_csv('C:/Users/Evan/PycharmProjects/StackNet/query_pred.csv', header=None)

pred_set = pd.DataFrame(data={'SK_ID_CURR': test['SK_ID_CURR'],
                              'TARGET': preds.loc[:, 1]})

pred_set.to_csv(output_path + time_now + '.csv', index=False)






