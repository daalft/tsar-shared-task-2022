from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

etc = ExtraTreesClassifier(n_estimators=400)
rfc = RandomForestClassifier(n_estimators=50)
bc = BaggingClassifier(n_estimators=200)
abc = AdaBoostClassifier()
mlp = MLPClassifier(hidden_layer_sizes=(20,20), activation="logistic")
dtc = DecisionTreeClassifier()
xgbc = xgb.XGBClassifier(eta=0.3, max_depth=5, n_estimators=100, objective="binary:logistic")
vc = VotingClassifier(estimators=[("etc", etc), ("rfc", rfc), ("bc", bc), ("abc", abc), ("mlp", mlp), ("dtc", dtc), ("xgb", xgbc)])

def load_data(fn):
    datas, labels = [], []
    with open(fn) as f:
        for l in f:
            if not l.strip():
                continue
            ps = l.rstrip().split("\t")
            data = [float(x) for x in ps[:-1]]
            label = int(ps[-1])
            datas.append(data)
            labels.append(label)
    return datas, labels

train_file = "/train_semeval_en.csv"
test_file = "/test_semeval_en.csv"
tsar_train = "/train_tsar_en.csv"
tsar_test = "/test_tsar_en.csv"
train_data, train_labels = load_data(train_file)
test_data, test_labels = load_data(test_file)
tsar_train_data, tsar_train_labels = load_data(tsar_train)
tsar_test_data, tsar_test_labels = load_data(tsar_test)
tsar_data = tsar_train_data + tsar_test_data
tsar_labels = tsar_train_labels + tsar_test_labels
full_data = train_data + tsar_train_data
full_label = train_labels + tsar_train_labels

vc.fit(full_data, full_label)
print("Semeval test score: " + vc.score(test_data, test_labels))
print("Tsar test score: " + vc.score(tsar_test_data, tsar_test_labels))
