from transformers import AutoModel, AutoTokenizer, BertModel
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

# GPU kullanılabilir mi diye kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Model ve tokenizer'ı yükle (large model)
model_large = AutoModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
tokenizer_large = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")



# Model ve tokenizer'ı yükle (small model)
model_small = BertModel.from_pretrained("ytu-ce-cosmos/turkish-small-bert-uncased")
tokenizer_small = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-small-bert-uncased")

# Model ve tokenizer'ı yükle (mini model)
model_mini = BertModel.from_pretrained("ytu-ce-cosmos/turkish-mini-bert-uncased")
tokenizer_mini = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-mini-bert-uncased")


# Model ve tokenizer'ı yükle (tiny model)
model_tiny = BertModel.from_pretrained("ytu-ce-cosmos/turkish-tiny-bert-uncased")
tokenizer_tiny = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-tiny-bert-uncased")

# Modeli GPU'ya taşı
model_large.to(device)
model_small.to(device)
model_mini.to(device)
model_tiny.to(device)


# 1.Metin Veri Kümesi

# Verileri eğiticili öğrenme için hazırla
df = pd.read_csv(r"C:\Users\BERKE\Desktop\archive\7allV03.csv")
texts = df['text'].tolist()
class_count = len(df['category'].unique())


label_mapping = {label: index for index, label in enumerate(df['category'].unique())}

# DataFrame'deki string etiketleri sayısal değerlere çevir
df['numeric_category'] = df['category'].map(label_mapping)

# Yeni sayısal etiketleri liste olarak al
labels = df['numeric_category'].tolist()




# 2.Metin Veri Kümesi

# Veri setinin bulunduğu dosya yolu
base_folder = r"C:\Users\BERKE\Desktop\dataset"
# Klasördeki alt klasörleri al
subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
class_names = [os.path.basename(subfolder) for subfolder in subfolders]
# Her bir alt klasördeki metin veri kümelerini yükle

texts = []
labels = []

for label, subfolder in enumerate(subfolders):
     files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(".txt")] 
     for file_path in files:
         with open(file_path, "r", encoding="utf-8") as file:
             text = file.read()
             texts.append(text)
             labels.append(class_names[label])
unique_values = list(set(labels))  # Dizideki unique değerleri bul

value_dict = {val: index for index, val in enumerate(unique_values)}  # Sözlük oluştur

labels = [value_dict[val] for val in labels]  # Her bir değeri sözlük kullanarak integer'a çevir

class_count = len(unique_values)




# 3.Metin Veri Kümesi
df = pd.read_csv(r"C:\Users\BERKE\Desktop\archive2\yorumsepeti.csv", sep=';')

df['speed'] = df['speed'].replace(['-'], np.nan)
df['service'] = df['service'].replace(['-'], np.nan)
df['flavour'] = df['flavour'].replace(['-'], np.nan)

df['speed'] = df['speed'].astype(float)
df['service'] = df['service'].astype(float)
df['flavour'] = df['flavour'].astype(float)
df['review'] = df['review'].astype(str)

df = df.assign(target=df.loc[:, ['speed', 'service', 'flavour']].mean(axis=1))
df['target'] = round(df['target'])

df = df.dropna(subset=['target'], axis=0)
df['target'] = df['target'].astype(int)

decode_map = { 1:"NEGATIVE", 2:"NEGATIVE" ,3: "NEGATIVE", 4:"NEGATIVE", 5:"NEGATIVE",6: "POSITIVE",
              7: "POSITIVE", 8: "POSITIVE", 9: "POSITIVE", 10: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

df.target = df.target.apply(lambda x: decode_sentiment(x))


texts = df["review"].tolist()


label_mapping = {label: index for index, label in enumerate(df["target"].unique())}

# DataFrame'deki string etiketleri sayısal değerlere çevir
df['numeric_category'] = df['target'].map(label_mapping)

# Yeni sayısal etiketleri liste olarak al
labels = df['numeric_category'].tolist()
class_count= len(df["target"].unique())







# 4.Metin Veri Kümesi
df = pd.read_csv(r"C:\Users\BERKE\Desktop\archive3\TurkishBookDataSet.csv")
Dataset = df.drop(["author","publisher","publication_year","pages_count","ISBN","book_img"], axis = 1)
Dataset = Dataset.drop([0])
Dataset = Dataset.dropna()
texts = Dataset['explanation'].tolist()
class_count = len(Dataset['book_type'].unique())

label_mapping = {label: index for index, label in enumerate(Dataset['book_type'].unique())}

# DataFrame'deki string etiketleri sayısal değerlere çevir
Dataset['numeric_category'] = Dataset['book_type'].map(label_mapping)

# Yeni sayısal etiketleri liste olarak al
labels = Dataset['numeric_category'].tolist()









train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

max_length = 128
batch_size = 8
#Tiny BERT token işlemi

train_tokens_tiny = tokenizer_tiny(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
test_tokens_tiny = tokenizer_tiny(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)


train_dataset_tiny = TensorDataset(train_tokens_tiny['input_ids'], train_tokens_tiny['attention_mask'], train_labels)
test_dataset_tiny = TensorDataset(test_tokens_tiny['input_ids'], test_tokens_tiny['attention_mask'], test_labels)

train_loader_tiny = DataLoader(train_dataset_tiny, batch_size=batch_size , shuffle=True)
test_loader_tiny = DataLoader(test_dataset_tiny, batch_size=batch_size , shuffle=False)


#Mini BERT token işlemi

train_tokens_mini = tokenizer_mini(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
test_tokens_mini = tokenizer_mini(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)


train_dataset_mini = TensorDataset(train_tokens_mini['input_ids'], train_tokens_mini['attention_mask'], train_labels)
test_dataset_mini = TensorDataset(test_tokens_mini['input_ids'], test_tokens_mini['attention_mask'], test_labels)

train_loader_mini = DataLoader(train_dataset_mini, batch_size=batch_size , shuffle=True)
test_loader_mini = DataLoader(test_dataset_mini, batch_size=batch_size , shuffle=False)

#Small BERT token işlemi

train_tokens_small= tokenizer_small(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
test_tokens_small= tokenizer_small(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)


train_dataset_small = TensorDataset(train_tokens_small['input_ids'], train_tokens_small['attention_mask'], train_labels)
test_dataset_small = TensorDataset(test_tokens_small['input_ids'], test_tokens_small['attention_mask'], test_labels)

train_loader_small = DataLoader(train_dataset_small, batch_size=batch_size , shuffle=True)
test_loader_small = DataLoader(test_dataset_small, batch_size=batch_size , shuffle=False)

#Large BERT token işlemi

train_tokens_large= tokenizer_large(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
test_tokens_large= tokenizer_large(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)


train_dataset_large = TensorDataset(train_tokens_large['input_ids'], train_tokens_large['attention_mask'], train_labels)
test_dataset_large = TensorDataset(test_tokens_large['input_ids'], test_tokens_large['attention_mask'], test_labels)

train_loader_large = DataLoader(train_dataset_large, batch_size=batch_size , shuffle=True)
test_loader_large = DataLoader(test_dataset_large, batch_size=batch_size , shuffle=False)



for param in model_tiny.parameters():
    param.requires_grad = True

for param in model_mini.parameters():
    param.requires_grad = True

for param in model_small.parameters():
    param.requires_grad = True

for param in model_large.parameters():
    param.requires_grad = True


num_layers_large = model_large.config.num_hidden_layers
num_layers_small = model_small.config.num_hidden_layers
num_layers_mini = model_mini.config.num_hidden_layers
num_layers_tiny = model_tiny.config.num_hidden_layers


classifiers_tiny = []
classifiers_mini = []
classifiers_small = []
classifiers_large = []

for i in range(num_layers_tiny):
    classifier_tiny = nn.Linear(model_tiny.config.hidden_size, class_count)
    model_tiny.encoder.layer[i].add_module(f'classifier_{i}', classifier_tiny)
    classifiers_tiny.append(classifier_tiny)


criterions_tiny = [nn.CrossEntropyLoss() for _ in range(num_layers_tiny)]
optimizers_tiny = [optim.Adam(classifier.parameters(), lr=1e-5) for classifier in classifiers_tiny]

for i in range(num_layers_mini):
    classifier_mini = nn.Linear(model_mini.config.hidden_size, class_count)
    model_mini.encoder.layer[i].add_module(f'classifier_{i}', classifier_mini)
    classifiers_mini.append(classifier_mini)


criterions_mini = [nn.CrossEntropyLoss() for _ in range(num_layers_mini)]
optimizers_mini = [optim.Adam(classifier.parameters(), lr=1e-5) for classifier in classifiers_mini]

for i in range(num_layers_small):
    classifier_small = nn.Linear(model_small.config.hidden_size, class_count)
    model_small.encoder.layer[i].add_module(f'classifier_{i}', classifier_small)
    classifiers_small.append(classifier_small)


criterions_small = [nn.CrossEntropyLoss() for _ in range(num_layers_small)]
optimizers_small = [optim.Adam(classifier.parameters(), lr=1e-5) for classifier in classifiers_small]

for i in range(num_layers_large):
    classifier_large = nn.Linear(model_large.config.hidden_size, class_count)
    model_large.encoder.layer[i].add_module(f'classifier_{i}', classifier_large)
    classifiers_large.append(classifier_large)


criterions_large = [nn.CrossEntropyLoss() for _ in range(num_layers_large)]
optimizers_large = [optim.Adam(classifier.parameters(), lr=1e-5) for classifier in classifiers_large]

# Bir tane Tiny sınıflandırıcı
classifier_tiny = nn.Linear(model_tiny.config.hidden_size, class_count).to(device)
criterion_tiny = nn.CrossEntropyLoss()
optimizer_tiny = optim.Adam(classifier_tiny.parameters(), lr=1e-5)
# Bir tane Mini sınıflandırıcı
classifier_mini = nn.Linear(model_mini.config.hidden_size, class_count).to(device)
criterion_mini = nn.CrossEntropyLoss()
optimizer_mini = optim.Adam(classifier_mini.parameters(), lr=1e-5)
# Bir tane Small sınıflandırıcı
classifier_small = nn.Linear(model_small.config.hidden_size, class_count).to(device)
criterion_small = nn.CrossEntropyLoss()
optimizer_small = optim.Adam(classifier_small.parameters(), lr=1e-5)
# Bir tane Large sınıflandırıcı
classifier_large = nn.Linear(model_large.config.hidden_size, class_count).to(device)
criterion_large = nn.CrossEntropyLoss()
optimizer_large = optim.Adam(classifier_large.parameters(), lr=1e-5)





# Eğitim döngüsü her katman için

def performance_all_layers(num_layer, model, classifiers, criterions, optimizers, train_loader, test_loader): 
    num_epochs = 10
    for epoch in range(num_epochs):

        train_corrects = [0 for _ in range(num_layer)]  # Her katmanın doğru tahmin sayısını depolamak için liste
        
        
        for i in range(num_layer):
            
            train_total = 0  # Toplam eğitim veri sayısı

            optimizers[i].zero_grad()

            for batch in tqdm(train_loader, desc='Eğitim Verisi'):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                
                layer_output = model(input_ids, attention_mask=attention_mask).last_hidden_state
               
                pooled_output = layer_output.mean(dim=1)     
                
                classifier = classifiers[i].to(device) 
                layer_logits = classifier(pooled_output)
                
                loss = criterions[i](layer_logits, labels)
                loss.backward()
                optimizers[i].step()
               
                # Accuracy hesapla
                preds = torch.argmax(layer_logits, dim=1)
                train_corrects[i] += torch.sum(preds == labels).item()
                train_total += len(labels)
                
          
            train_accuracy = train_corrects[i] / train_total
            print(f"Epoch {epoch + 1}/{num_epochs}, Layer {i + 1}, Train Accuracy: {train_accuracy:.4f}")



            
        #Test Performansını hesaplama
        
        with torch.no_grad(): 
            test_corrects = [0 for _ in range(num_layer)]  
            
            for i in range(num_layer):
                test_total = 0  
                
                for batch in tqdm(test_loader, desc='Test Verisi'):
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                   
                    
                    layer_output = model(input_ids, attention_mask=attention_mask).last_hidden_state
                   
                    pooled_output = layer_output.mean(dim=1)
                    
                   
                    classifier = classifiers[i].to(device)
                    layer_logits = classifier(pooled_output)
                    
                  
                    # Accuracy hesapla
                    preds = torch.argmax(layer_logits, dim=1)
                    test_corrects[i] += torch.sum(preds == labels).item()
                    test_total += len(labels)
                    

                
    
                test_accuracy = test_corrects[i] / test_total
                print(f"Epoch {epoch + 1}/{num_epochs}, Layer {i + 1}, Test Accuracy: {test_accuracy:.4f}")





performance_all_layers(num_layers_tiny, model_tiny, classifiers_tiny, criterions_tiny, optimizers_tiny,train_loader_tiny,test_loader_tiny)
performance_all_layers(num_layers_mini, model_mini, classifiers_mini, criterions_mini, optimizers_mini,train_loader_mini,test_loader_mini)
performance_all_layers(num_layers_small, model_small, classifiers_small, criterions_small, optimizers_small,train_loader_small,test_loader_small)
performance_all_layers(num_layers_large, model_large, classifiers_large, criterions_large, optimizers_large,train_loader_large,test_loader_large)

def train_and_evaluate_merged_layer(model, classifier, criterion, optimizer, train_loader, test_loader):
    num_epochs = 10
    for epoch in range(num_epochs):
        classifier.train()
        train_corrects = 0
        train_total = 0
        optimizer.zero_grad()

        for batch in tqdm(train_loader, desc='Eğitim Verisi'):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

           
            layer_outputs_train = model(input_ids, attention_mask=attention_mask).last_hidden_state
            
            merged_train_output = layer_outputs_train.mean(dim=1)

          
            layer_logits = classifier(merged_train_output)
            
            loss = criterion(layer_logits, labels)
            loss.backward()
            optimizer.step()

            # Accuracy hesapla
            preds = torch.argmax(layer_logits, dim=1)
            train_corrects += torch.sum(preds == labels).item()
            train_total += len(labels)

        train_accuracy = train_corrects / train_total
        print(f"Epoch {epoch + 1}/{num_epochs}, Merged Layer Train Accuracy: {train_accuracy:.4f}")

        #Test Performansını hesaplama
        classifier.eval()  
        with torch.no_grad(): 
            test_corrects = 0
            test_total = 0

            for batch in tqdm(test_loader, desc='Test Verisi'):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                
                layer_outputs_test = model(input_ids, attention_mask=attention_mask).last_hidden_state
               
                merged_test_output = layer_outputs_test.mean(dim=1)

               
                layer_logits = classifier(merged_test_output)
                
                # Accuracy hesapla
                preds = torch.argmax(layer_logits, dim=1)
                test_corrects += torch.sum(preds == labels).item()
                test_total += len(labels)

            test_accuracy = test_corrects / test_total
            print(f"Epoch {epoch + 1}/{num_epochs}, Merged Layer Test Accuracy: {test_accuracy:.4f}")


train_and_evaluate_merged_layer(model_tiny, classifier_tiny, criterion_tiny, optimizer_tiny, train_loader_tiny, test_loader_tiny)
train_and_evaluate_merged_layer(model_mini, classifier_mini, criterion_mini, optimizer_mini, train_loader_mini, test_loader_mini)
train_and_evaluate_merged_layer(model_small, classifier_small, criterion_small, optimizer_small, train_loader_small, test_loader_small)
train_and_evaluate_merged_layer(model_large, classifier_large, criterion_large, optimizer_large, train_loader_large, test_loader_large)

