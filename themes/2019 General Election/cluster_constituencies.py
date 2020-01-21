from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA

file = r'vote_counts.csv'
df = pd.read_csv(file)
tolerance = 10


important_columns = ['ONSConstID','CON','LAB','LIBDEM','GRN','SNP', 'PC', 'BXP', 'UKIP', 'OTHER', 'Registered Voters']
constituencies = df['Constituency'].to_numpy()
IDs = df['ONSConstID'].to_numpy()
registered_voters = pd.to_numeric(df['Registered Voters']).to_numpy()

votes = np.nan_to_num(df[important_columns[1:-1]].to_numpy())
vote_vectors = np.divide(votes, registered_voters[:,None])

colours = ['blue', 'red', 'gold', 'green', 'brown', 'green', 'light blue', 'purple', 'gray']
winners = np.argmax(vote_vectors, axis=1)

winner_colours=[]
for winner in winners:
    winner_colours.append(colours[winner])


def PCA_2D():
    pca = PCA(n_components=2)
     
    principalComponents = pca.fit_transform(vote_vectors)
    fig, ax = plt.subplots()
    ax.scatter(principalComponents[:,0], principalComponents[:,1], picker=tolerance, c=winner_colours)
    # for i, txt in enumerate(constituencies):
    #     ax.annotate(txt, (principalComponents[i,0], principalComponents[i,1]))
     
    def on_pick(event):
        ind = event.ind
        print(np.take(constituencies, ind))
     
    fig.canvas.callbacks.connect('pick_event', on_pick)
    
    plt.show()

def PCA_3D():
    pca = PCA(n_components=3)
    
    principalComponents = pca.fit_transform(vote_vectors)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.scatter(principalComponents[:,0], principalComponents[:,1],principalComponents[:,2], picker=tolerance, c=winner_colours)
    # for i, txt in enumerate(constituencies):
    #     ax.annotate(txt, (principalComponents[i,0], principalComponents[i,1]))
    
    def on_pick(event):
        ind = event.ind
        print(np.take(constituencies, ind))
    
    fig.canvas.callbacks.connect('pick_event', on_pick)
    
    plt.show()

PCA_2D()
PCA_3D()

