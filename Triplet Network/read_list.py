import pprint, pickle

with open('1.pkl', 'rb') as f:

    data1 = pickle.load(f)
    print(data1)
    print(len(data1))

with open('2.pkl', 'rb') as f:

    data2 = pickle.load(f)
    print(data2)
    print(len(data2))