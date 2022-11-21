"""# START REGION: PREPROCESS"""
import pandas as pd
from preprocess_util import process_url
import random
from pathlib import Path
import time
def train_preprocess(data_path, model_type="sequential", feature_set='full_set', test_size=0.3, random_state=random.randrange(100), number_records=100):
    """Check xem có phải file và đuôi csv không -> đọc file"""
    if Path(data_path).is_file() and Path(data_path).suffix == '.csv':
        data = pd.read_csv(data_path,sep=',').sample(n = 100, random_state = 2)
    """Kiểm tra xem chọn feature set nào"""
    if feature_set=='full_set':
        """Trích xuất features, bổ sung vào dataset ban đầu"""
        url_dataset = process_url(data)

        """Bỏ cột thừa"""

        one_hot = pd.get_dummies(url_dataset['protocol'])

        url_dataset.drop(['protocol'],axis=1, inplace=True)

        url_dataset = url_dataset.join(one_hot, how = 'left', lsuffix='_left', rsuffix='_right')

        url_dataset.drop('Unnamed: 0',axis=1,inplace=True)
        #one_hot = pd.get_dummies(url_dataset['registrar'])
        #url_dataset.drop('registrar',axis=1,inplace=True)
        #url_dataset = url_dataset.join(one_hot)

        """Tính correlation để bỏ bớt các columns"""
        if False:
            url_dataset.pivot("url_length", "Label", "val")

            corr_mat =url_dataset.corr()
            corr_mat

            url_dataset.info()

            for x in range(0,len(corr_mat)-1,1):
                corr_mat.iloc[x,x]=0
                
            s=corr_mat['Label'].abs().sort_values()
            s=s[s>0.005]
            s.index

            fig, ax = plt.subplots(figsize=(20,20))   
            sns.heatmap(url_dataset.corr(), annot=True, linewidths=.5, ax=ax)

        """Bỏ cột tạm chưa scale được"""
        column_name = [col for col in url_dataset.columns if col not in ['Label','url_length','url','subdomain', 'domain', 'tld','word_process','today_date','creation_date',
            'expiration_date', 'last_updated', 'registrar']]
        #column_name = [col for col in url_dataset.columns if col in s.index]
        x_dataset = url_dataset[column_name]
        y_dataset = url_dataset['Label']
    """Scale dữ liệu về khoảng [0,1]"""
    from sklearn.preprocessing import RobustScaler

    column_names =x_dataset.columns
    scaler =  RobustScaler(quantile_range=(25, 75))
    x_dat = scaler.fit_transform(x_dataset)

    x_dataset =pd.DataFrame(x_dat,columns=column_names)

    # from sklearn.preprocessing import MinMaxScaler

    # column_names =x_dataset.columns
    # scaler = MinMaxScaler()
    # x_dat = scaler.fit_transform(x_dataset)

    # x_dataset =pd.DataFrame(x_dat,columns=column_names)

    """Chia tập train, test"""
    from sklearn.model_selection import train_test_split    

    X_train,X_test,y_train,y_test=train_test_split(x_dataset,y_dataset,test_size=test_size,random_state=random_state)
    print(X_train)
    message= "Train size {} samples.".format(X_train.shape[0])+ "Test size {} samples.".format(X_test.shape[0]) 

    return  {
                'X_train':X_train,
                'X_test':X_test,
                'y_train':y_train,
                'y_test':y_test,
                'model_type':model_type,
                'message':message
            }

def inference_preprocess(data_path,feature_set='full_set'):
    start_time=time.time()
    """Check xem có phải file và đuôi csv không -> đọc file"""
    if Path(data_path).is_file() and Path(data_path).suffix == '.csv':
        data = pd.read_csv(data_path,sep=',')#.sample(n = 100, random_state = 2)
    """Kiểm tra xem chọn feature set nào"""
    if feature_set=='full_set':
        """Trích xuất features, bổ sung vào dataset ban đầu"""
        url_dataset = process_url(data)

        """Bỏ cột thừa"""

        one_hot = pd.get_dummies(url_dataset['protocol'])

        url_dataset.drop(['protocol'],axis=1, inplace=True)

        url_dataset = url_dataset.join(one_hot, how = 'left', lsuffix='_left', rsuffix='_right')

        url_dataset.drop('Unnamed: 0',axis=1,inplace=True)
        #one_hot = pd.get_dummies(url_dataset['registrar'])
        #url_dataset.drop('registrar',axis=1,inplace=True)
        #url_dataset = url_dataset.join(one_hot)

        
        """Bỏ cột tạm chưa scale được"""
        column_name = [col for col in url_dataset.columns if col not in ['Label','url_length','url','subdomain', 'domain', 'tld','word_process','today_date','creation_date',
            'expiration_date', 'last_updated', 'registrar']]
        #column_name = [col for col in url_dataset.columns if col in s.index]
        x_dataset = url_dataset[column_name]
    """Scale dữ liệu về khoảng [0,1]"""
    from sklearn.preprocessing import RobustScaler

    column_names =x_dataset.columns
    scaler =  RobustScaler(quantile_range=(25, 75))
    x_dat = scaler.fit_transform(x_dataset)

    x_dataset =pd.DataFrame(x_dat,columns=column_names)

    from sklearn.preprocessing import MinMaxScaler

    column_names =x_dataset.columns
    scaler = MinMaxScaler()
    x_dat = scaler.fit_transform(x_dataset)

    x_dataset =pd.DataFrame(x_dat,columns=column_names)
    end_time=time.time()
    """Chia tập train, test"""    
    message= "Input data size: {} samples".format(x_dataset.shape[0])+'\nPreprocess time: %2.1fs' %(end_time-start_time)
    
    return  {
                'dataset':x_dataset,
                'message':message
            }
"""# END REGION: PREPROCESS"""