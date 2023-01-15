import os
path = '/Users/zhenjiaodu/no_icloud/1. ksu-in class/CIS 730 introduction to artifical intelligence/term project/BERT_4_ACE/uber_rides_prediction_using_machine_learning'
os.chdir(path)

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math
import collections
from keras.models import load_model
from flask import Flask,request, url_for, redirect, render_template
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
# model = pickle.load(open('model.pkl','rb'))
# load the prediction model
model = load_model('DPPIV_tensorflow_model')
# embeddings function
def esm_embeddings(peptide_sequence_list: list):
    # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
    #         or you have too many sequences for transformation in a single converting,
    #         you conputer might automatically kill the job.
    # return a panda.dataframe
    import torch
    import pandas as pd
    import esm
    import collections
    # load the model
    # NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # load the peptide sequence list into the bach_converter
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    ## batch tokens are the embedding results of the whole data set

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        # Here we export the last layer of the EMS model output as the representation of the peptides
       # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    # save dataset
    # sequence_representations is a list and each element is a tensor
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
        # tensor can be transformed as numpy sequence_representations[0].numpy() or  sequence_representations[0].to_list
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    return embeddings_results

# normalized the embeddings
X_train_data_name = 'DPPIV_train_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_train_data = pd.read_csv(X_train_data_name, header=0, index_col=0, delimiter=',')
X_train = np.array(X_train_data)
# normalize the X data range
scaler = MinMaxScaler()
scaler.fit(X_train)
# scaler.transform will automatically transform the pd.dataframe into a np.array data format

# collect the output
def assign_activity(predicted_class):
    import collections
    out_put = []
    for i in range(len(predicted_class)):
        if predicted_class[i] == 0:
            #out_put[int_features[i]].append(1)
            out_put.append('active')
        else:
            #out_put[int_features[i]].append(2)
            out_put.append('non-active')
    return out_put

# create an app object using the Flask class
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 每一个网页上的 输入的框，是一个单独的x，下面这个就是吧这个单独的信息变成一个list，每一个单独的就是一个str （也可以吧x变成int 如果想要的话）
    # int_features  = [str(x) for x in request.form.values()] # this command basically use extract all the input into a list
    #final_features = [np.array(int_features)]
    int_features  = [str(x) for x in request.form.values()]
    sequence_list=int_features[0].split(',')  # 因为这个list里面只有一个element，所以我只需要把吧这个拿出来，然后split
    # 另外需要注意，这个地方，网页上输入的时候必须要是AAA,CCC,SAS, 这个格式，不同的sequence的区分只能使用逗号，其他的都不可以
    peptide_sequence_list = []
    for seq in sequence_list:
        format_seq = [seq, seq]  # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list.append(tuple_sequence)  # build a summarize list variable including all the sequence information

    embeddings_results = esm_embeddings(peptide_sequence_list) #conduct the embedding
    normalized_embeddings_results = scaler.transform(embeddings_results) # normalized the embeddings

    # prediction
    predicted_protability = model.predict(normalized_embeddings_results, batch_size=1)
    predicted_class = []
    for i in range(predicted_protability.shape[0]):
        index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
        predicted_class.append(index) # get the class of the results
    predicted_class = assign_activity(predicted_class) # transform results (0 and 1) into 'active' and 'non-active'

    return render_template('index.html',prediction_text="Number of Weekly Rides Should be {}".format(predicted_class))

if __name__ == '__main__':
    app.run()
