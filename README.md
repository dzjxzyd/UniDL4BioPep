# UniDL4BioPep
The implementation of the paper  [Du, Z., Ding, X., Xu, Y., & Li, Y. (2023).UniDL4BioPep: a universal deep learning architecture for binary classification in peptide bioactivity. Briefings in Bioinformatics, bbad135.](https://www.researchgate.net/publication/369832351_UniDL4BioPep_a_universal_deep_learning_architecture_for_binary_classification_in_peptide_bioactivity)
webserver available at [server_link](https://nepc2pvmzy.us-east-1.awsapprunner.com/)

Updates: Xingjian Ding release the PyTorch version of [UniDL4BioPep-ASL](https://github.com/David-Dingle/UniDL4BioPep_ASL_PyTorch) for imbalanced dataset. This method employed another loss function (asymmetric loss function, modified version of Focal loss function), which has the ability to conduct tunning both positive and negative sides at the same time.


2024-11-21 updates: all the datasets used in this study are uploaded into the corresponding folders and label 1 means the positive in this properties (for example: 1 is toxic, 0 is non-toxic; 1 is allergenic and 0 is non-allergenic; 1 is bitter and 0 is non-bitter; 1 is antimicrobial and 0 is non-antimicrobial, etc.) The training and test datasets are following the original dataset division of our referneces, if there is only one dataset inside the folder, that means the dataset division was not provided by the original refernece. 
Beyond the original 20 datasets, we add two more datasets (allergenic protein and peptides, and Cell penetrating peptides), both models are also developed with UniDL4BioPep model architecture and available at our webserver. 


2024-01-06 updates: we add the protability information to the prediction results, you will get both active&non-active and a protability (0.98) to indicate the probability our model predict. Allow you to be more easier to access to the model's results. (Please upload your file and then make prediction, you will get the new features) ! 

2023-07-04 updates: we re-design the template file (Pretrained_model_usage_template.ipynb). Now, it  can automatically use your GPU resource for peptide embedding and model prediction acceleration. Thanks for any feedback in this projects. 

2023-05-07 updates: we add a new-designed template (GPU_UniDL4BioPep_template_for_other_bioactivity.ipynb). It can automatically recognize your GPU if available and use your GPU for peptide embeddings and model fitting acceleration. Also, add a section for the fasta.format file transformation to csv files. 

Updates:  We add an advanced version (UniDL4BioPep-FL) employing focal loss function for imbalanced dataset and a template for your usage (UniDL4BioPep_FL_template_for_other_bioactivity.ipynb).

Usage of **UniDL4BioPep-FL** for **Imbalanced dataset**: Please select your minority group as a positive group (labeled as 1) and majority group as a negative group (labeled as 0); suggestions for hyperparameter tunning: gamma(0,1,2,3,4,5) and pos_weight (,0.1,0.2,...1.0) or no need to specify pos_weight. 

Notice: The model can also be used for multiclass classification (we adopt the softmax function at the last output layer), so you can just simply change the output layer node numbers. (feel free to reach out to me or submit your question in the Issue part.)

Updates: The web server with the advanced 26 models is available at https://nepc2pvmzy.us-east-1.awsapprunner.com/; the Webserver development repository is available at [UniDL4BioPep_web_server](https://github.com/dzjxzyd/UniDL4BioPep_web_server). 

**Notice: UniDL4BioPep is ONLY freely available for academic research; for commercial usage, please contact us**, zhenjiao@ksu.edu; xjding@ksu.edu; yonghui@ksu.edu;

If the contents are useful to you, Please kindly **Star** it and **Cite** it.
Please cite as: _Du, Z., Ding, X., Xu, Y., & Li, Y. (2023).UniDL4BioPep: a universal deep learning architecture for binary classification in peptide bioactivity. Briefings in Bioinformatics, bbad135._


## Requirements
The majoy dependencies used in this project are as following:
```
Python 3.8.16
fair-esm 2.0.0
keras 2.9.0
pandas 1.3.5
numpy 1.21.6
scikit-learn 1.0.2
tensorflow 2.9.2
torch 1.13.0+cu116
focal-loss
```
More detailed python libraries used in this project are referred to ```requirements.txt```. 
All the implementation can be down in Google Colab and all you need is just a browser and a google account.
Install all the above packages by ```!pip install package_name==2.0.0```


## Usage
Notice: all my dataset use 0 and 1 to represent positive and negative, respectively. Again, 0 is positive and 1 is negative.
### Use the pretrained model for your own dataset

Just check the file **Pretrained_model_usage_template.ipynb**

All you need is to prepare your data for prediction in a xlsx format file and open **Pretrained_model_usage_template.ipynb** in Google Colab.
Then upload your data and train dataset (for the model training). 
Then you are ready to go. 

### Train your own model with UniDL4BioPep

All you need to do is to prepare your databasets in a xlsx format and two column (first column is sequence and the second column is label).
You can just download the xlsx format dataset file from any folder in this repository. Before loading your dataset, please shuffle your datasets and split them as a train dataset and a test datasets as your requirement.

You can also use split dataset in python code with the following codes, and then you can replase the **data loading and embeddings** section anymore. Just replace that part with the following codes. 

UPDATES: I have add a new section in **UniDL4BioPep_template_for_other_bioactivity.ipynb** to fit you one xlsx format dataset loading and embeddings (just use it).
```
import numpy as np
import pandas as pd
# whole dataset loading and dataset splitting 
dataset = pd.read_excel('whole_sample_dataset.xlsx',na_filter = False) # take care the NA sequence problem

# generate the peptide embeddings
sequence_list = dataset['sequence'] 
embeddings_results = pd.DataFrame()
for seq in sequence_list:
    format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
    tuple_sequence = tuple(format_seq)
    peptide_sequence_list = []
    peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
    # employ ESM model for converting and save the converted data in csv format
    one_seq_embeddings = esm_embeddings(peptide_sequence_list)
    embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])
embeddings_results.to_csv('whole_sample_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv')

# loading the y dataset for model development 
y = dataset['label']
y = np.array(y) # transformed as np.array for CNN model

# read the peptide embeddings
X_data_name = 'whole_sample_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_data = pd.read_csv(X_data_name,header=0, index_col = 0,delimiter=',')
X = np.array(X_data)

# split dataset as training and test dataset as ratio of 8:2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)

# normalize the X data range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) # normalize X to 0-1 range 
X_test = scaler.transform(X_test)
```
After the transoformation, you are all set and good to go. 
Notice: please do check your dataset dimension before running in case of error occring.
```
# check the dimension of the dataset before model development
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```
### Further model tuning and modifications

Feel free to make your personalized modifications. Just scroll down to the model architecture sections and make revisions to fit your expectation.

In my experiments, this architecture seems quite good and you might need to take a big change to make something different if you want. 




