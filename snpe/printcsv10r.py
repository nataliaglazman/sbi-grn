import pandas as pd
column_name = ['k1','k2','k3']
cnn1_2r1k3p = pd.DataFrame(data=posterior_samples_cnn1, columns=column_name)
cnn1_2r1k3p.to_csv('cnn1_10r1500s3p.csv')