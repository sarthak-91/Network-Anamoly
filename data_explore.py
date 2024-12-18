import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
# Set TensorFlow to use multiple threads
#num_threads = 7  # Use 7 threads
#config = tf.compat.v1.ConfigProto(
#    intra_op_parallelism_threads=num_threads,
#    inter_op_parallelism_threads=num_threads
#)
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)

# Column names for NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


# Convert TXT to CSV and preprocess the data
def load_and_preprocess_data(csv_path, kind='binary'):
    # Load TXT file
    data = pd.read_csv(csv_path)

    # Encode categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
    # Convert labels to binary (normal = 0, anomaly = 1)
    print(data['label'].unique())
    if kind =='binary':data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
    # Scale numerical features
    
    X = data.drop(['label', 'difficulty'], axis=1)
    # Replace original data with scaled data

    # Separate features and labels
    y = data['label'].values
    # Balance dataset using SMOTE
    # Split into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=42)
    #smote = SMOTE(random_state=42)
    #x_train,y_train= smote.fit_resample(x_train, y_train)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train=to_categorical(y_train)
    y_test = to_categorical(y_test)

    os.makedirs(kind,exist_ok=True)
    np.savetxt(f"{kind}/train_x.txt",x_train)
    np.savetxt(f"{kind}/test_x.txt",x_test)
    np.savetxt(f"{kind}/train_y.txt",y_train)
    np.savetxt(f"{kind}/test_y.txt",y_test)
    return x_train,x_test,y_train,y_test

if __name__ == "__main__":
    csv_path = "traffic.csv"
    load_and_preprocess_data(csv_path)
