import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from utils_fluoroscopy import compute_embedding, compute_kl_feat, load_model, load_model2

def get_embedding(testsets, kl_path, which_embedding, device):
    # train_sample=160 
    # valid_sample=20
    train_sample=2 
    valid_sample=2 
    train_embeddings           = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_train.pt')[:train_sample]
    valid_embeddings           = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_valid.pt')[:valid_sample]
    test_embeddings            = torch.load(f'save/{which_embedding}_embedding/save_embedding/HC3_en_test.pt')[:valid_sample]
            
    train_labels               = torch.load('dataset/labels/HC3_en_train.pt')[:train_sample].to(device)
    valid_labels               = torch.load('dataset/labels/HC3_en_valid.pt')[:valid_sample].to(device)
    test_labels                = torch.load('dataset/labels/HC3_en_test.pt')[:valid_sample].to(device)
    
    train_embeddings = train_embeddings.to(device)
    valid_embeddings = valid_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)

    with open(f'save/{kl_path}/HC3_en_train.pkl', 'rb') as f:
        train_kl = pickle.load(f)
        train_kl = np.array(train_kl)
        idx = train_kl.argmax(axis=1)
        if which_layer == 'max_kl':
            train_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings,idx)]).to(device)
        if which_layer == 'max_kl_and_last_layer':
            train_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(train_embeddings,idx)]),
                                          train_embeddings[:,-embedding_dim:]], dim=1).to(device)

    with open(f'save/{kl_path}/HC3_en_test.pkl', 'rb') as f:
        test_kl = pickle.load(f)
        test_kl = np.array(test_kl)
        idx = test_kl.argmax(axis=1)
        if which_layer == 'max_kl':
            test_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(test_embeddings,idx)]).to(device)
        if which_layer == 'max_kl_and_last_layer':
            test_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(test_embeddings,idx)]),
                                         test_embeddings[:,-embedding_dim:]], dim=1).to(device)

    with open(f'save/{kl_path}/HC3_en_valid.pkl', 'rb') as f:
        valid_kl = pickle.load(f)
        valid_kl = np.array(valid_kl)
        idx = train_kl.argmax(axis=1)
        if which_layer == 'max_kl':
            valid_embeddings = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(valid_embeddings,idx)]).to(device)
        if which_layer == 'max_kl_and_last_layer':
            valid_embeddings = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row, i in zip(valid_embeddings,idx)]),
                                          valid_embeddings[:,-embedding_dim:]], dim=1).to(device)

    if which_layer == 'first_layer':
        train_embeddings = train_embeddings[:,:embedding_dim].to(device)
        valid_embeddings = valid_embeddings[:,:embedding_dim].to(device)
        test_embeddings = test_embeddings[:,:embedding_dim].to(device)
    elif which_layer == 'last_layer':
        train_embeddings = train_embeddings[:,-embedding_dim:].to(device)
        valid_embeddings = valid_embeddings[:,-embedding_dim:].to(device)
        test_embeddings = test_embeddings[:,-embedding_dim:].to(device)
    elif which_layer == 'first_and_last_layers':
        train_embeddings = torch.cat([train_embeddings[:,:embedding_dim],train_embeddings[:,-embedding_dim:]], dim=1).to(device)
        valid_embeddings = torch.cat([valid_embeddings[:,:embedding_dim],valid_embeddings[:,-embedding_dim:]], dim=1).to(device)
        test_embeddings = torch.cat([test_embeddings[:,:embedding_dim],test_embeddings[:,-embedding_dim:]], dim=1).to(device)
    elif which_layer.startswith('layer_'):
        if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
            layer_num = int(which_layer.split('_')[-1])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
        elif 'last_layer' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            train_embeddings = torch.cat([train_embeddings[:,-embedding_dim:],train_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
            valid_embeddings = torch.cat([valid_embeddings[:,-embedding_dim:],valid_embeddings[:,(layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
        elif 'later_layer' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:].to(device)    
        elif 'to' in which_layer:
            layer_num = int(which_layer.split('_')[1])
            layer_num2 = int(which_layer.split('_')[3])
            train_embeddings = train_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)
            valid_embeddings = valid_embeddings[:,(layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)

    num = 2
    testset_embeddings = []
    testset_labels = []
    for file_name in testsets:
        testset_embeddings.append(torch.load(f'save/{which_embedding}_embedding/save_embedding/{file_name}')[:num])
        testset_labels.append(torch.load(f'dataset/labels/{file_name}').to(device)[:num])
        with open(f'save/{kl_path}/{file_name.split(".")[0]}.pkl', 'rb') as f:
            kl = pickle.load(f)
            kl = np.array(kl)
            idx = kl.argmax(axis=1)
            if which_layer == 'max_kl':
                testset_embeddings[-1] = torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(testset_embeddings[-1], idx)]).to(device)
            elif which_layer == 'max_kl_and_last_layer':
                testset_embeddings[-1] = torch.cat([torch.tensor([row[(i+1)*embedding_dim:(i+2)*embedding_dim].tolist() for row ,i in zip(testset_embeddings[-1],idx) ]),
                                                    testset_embeddings[-1][:,-embedding_dim:]],dim=1).to(device)
            elif which_layer == 'first_layer':
                testset_embeddings[-1] = testset_embeddings[-1][:, :embedding_dim].to(device)
            elif which_layer == 'last_layer':
                testset_embeddings[-1] = testset_embeddings[-1][:, -embedding_dim:].to(device)
            elif which_layer == 'first_and_last_layers':
                testset_embeddings[-1] = torch.cat([testset_embeddings[-1][:, :embedding_dim], testset_embeddings[-1][:, -embedding_dim:]], dim=1).to(device)
            elif which_layer.startswith('layer_'):
                if 'last_layer' not in which_layer and 'later_layer' not in which_layer and 'to' not in which_layer:
                    layer_num = int(which_layer.split('_')[-1])
                    testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim].to(device)
                elif 'last_layer' in which_layer:
                    layer_num = int(which_layer.split('_')[1])
                    testset_embeddings[-1] = torch.cat([testset_embeddings[-1][:, -embedding_dim:], testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num+1)*embedding_dim]], dim=1).to(device)
                elif 'later_layer' in which_layer:
                    layer_num = int(which_layer.split('_')[1])
                    testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:].to(device)
                elif 'to' in which_layer:
                    layer_num = int(which_layer.split('_')[1])
                    layer_num2 = int(which_layer.split('_')[3])
                    testset_embeddings[-1] = testset_embeddings[-1][:, (layer_num)*embedding_dim:(layer_num2+1)*embedding_dim].to(device)
    
    return train_embeddings, train_labels, valid_embeddings, valid_labels, testset_embeddings, testset_labels

def test(model, test_set, test_label, test_acc):
    with torch.no_grad():
        outputs = model(test_set)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        auroc = roc_auc_score(test_label.cpu().numpy(), probabilities.cpu().numpy())
        test_acc.append(auroc)
    return auroc

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=2, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.classifier(x)
        return x
    

def train(train_embeddings, train_labels, hidden_sizes, droprate, device, 
          valid_embeddings=None, valid_labels=None, testset_embeddings=None, testset_labels=None):
    input_size = train_embeddings.shape[1]
    model = BinaryClassifier(input_size,hidden_sizes=hidden_sizes,dropout_prob=droprate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 10
    batch_size = 16
    best_valid_acc = 0
    best_test_acc = 9999.9
    for epoch in range(num_epochs):
        for i in range(0, len(train_embeddings), batch_size):
            model.train()
            batch_embeddings = train_embeddings[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if valid_embeddings is not None and valid_labels is not None:
            model.eval()
            with torch.no_grad():
                outputs = model(valid_embeddings)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == valid_labels).sum().item() / len(valid_labels)
                testset_acc = []
                for test_embed, test_label in zip(testset_embeddings, testset_labels):
                    test(model, test_embed, test_label, testset_acc)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}, Xsum/writing/pub/gpt4Xsum/gpt4writing/gpt4pub/claude3Xsum/claude3writing/claude3pub Test auroc: {','.join([str(round(i,4)) for i in testset_acc])}")
                if accuracy > best_valid_acc:
                    best_valid_acc = accuracy
                    best_test_acc = testset_acc
    return best_test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--droprate', type=float, default=0.4)
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--which_layer', type=str, default="max_kl")
    parser.add_argument('--which_embedding', type=str, default='gte-qwen_all')
    parser.add_argument('--kl_path', type=str, default='gte-qwen_KL_with_first_and_last_layer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_dataset', type=str, default="./exp_main/data/xsum_gpt-neo-2.7B")
    parser.add_argument('--output_file', type=str, default="./exp_main/results/")
    parser.add_argument('--model_name', type=str, default="Alibaba-NLP/gte-Qwen1.5-7B-instruct")
    parser.add_argument('--cache_dir', type=str, default='../../../cache')
    parser.add_argument('--train_dataset', type=str, default='./exp_main/data/squad_gpt-neo-2.7B')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    learning_rate=args.lr
    droprate=args.droprate
    embedding_dim=args.embedding_dim
    which_layer=args.which_layer
    which_embedding=args.which_embedding
    kl_path=args.kl_path
    max_length=args.max_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = args.model_name
    cache_dir = args.cache_dir
    # cache_dir = '/root/autodl-tmp/cache'
    
    fl_layer_dir = 'gte-qwen_KL_with_first_and_last_layer'
    save_dir = f'save/{fl_layer_dir}/'
    data_dir = 'dataset/processed_data/'
    tokenizer, model = load_model(model_name, cache_dir)
    compute_kl_feat(model, tokenizer, data_dir, save_dir, max_length, device)
    
    all_embedding_dir='gte-qwen_all_embedding'
    save_dir = f'save/{all_embedding_dir}/save_embedding/'
    data_dir = 'dataset/processed_data/'
    tokenizer, model = load_model2(model_name, cache_dir)
    compute_embedding(model, tokenizer, data_dir, save_dir, max_length, device)
    
    testsets = ['Xsum_gpt3.pt', 'writing_gpt-3.pt', 'pub_gpt-3.pt', 
                'gpt4-Xsum-gpt3.pt', 'gpt4-writing-gpt3.pt', 'gpt4-pub-gpt3.pt', 
                'xsum_claude-3-opus-20240229-gpt3.pt', 'writing_claude-3-opus-20240229-gpt3.pt', 'pub_claude-3-opus-20240229-gpt3.pt',]

    train_embeddings, train_labels, valid_embeddings, valid_labels, testset_embeddings, test_labels = get_embedding(testsets, kl_path, which_embedding, device)

    clf_hidden_dim = [1024, 512]
    best_test_acc = train(train_embeddings, train_labels, clf_hidden_dim, droprate, device, 
                          valid_embeddings, valid_labels, testset_embeddings, test_labels) 
    print('='*20)
    print('best test acc:', best_test_acc)
    # print('best test acc:', ','.join([str(round(i,4)) for i in best_test_acc]))
    # print('average test acc:', [sum(best_test_acc[i*3:i*3+3])/3 for i in range(3)])