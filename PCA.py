import numpy as np
import matplotlib.pyplot as plt
data=np.array([[2,3],[3,4],[4,5],[5,6],[6,7]])

mean=np.mean(data,axis=0)
sd=np.std(data,axis=0)
ds=(data-mean)/sd

cm=np.cov(ds.T)
eigv ,eigve=np.linalg.eig(cm)

pc=eigve[:,np.argmax(eigv)]

pm=pc.reshape(-1,1)

pca_data=ds.dot(pm)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.scatter(ds[:,0],ds[:,1])
plt.title('Original Data (2D)')
plt.xlabel('Feature1')

plt.ylabel('Feature2')

plt.subplot(1,2,2)
plt.scatter(pca_data,np.zeros_like(pca_data))

plt.title('PCA -transfomed Data(1D)')
plt.xlabel('Principal Component')
plt.ylabel(' ')
plt.tight_layout()
plt.show()
