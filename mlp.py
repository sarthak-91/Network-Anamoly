from network_class import NeuralNetworkManager
from useful import load_from_file, print_scores



if __name__ == "__main__":
    x,y = load_from_file(folder='binary',set="train")
    mlp = NeuralNetworkManager("mlp classify", x_train_ndim=x.shape[1],folder='mlp',learning_rate=0.001)
    mlp.create_network([20,5],output_num=2,output_actiavtion='softmax')
    mlp.compile_model(loss='categorical_crossentropy',eval_metric='accuracy')
    mlp.model.summary()
    mlp.fit_model(x,y,epochs=50,batch_size=10)
    x,y = load_from_file(folder='binary',set='test')
    y_pred = mlp.model.predict(x).argmax(axis=1)
    print_scores(y.argmax(axis=1),y_pred)