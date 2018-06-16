from model import *
from prepare_dataset import *

from prepare_dataset_for_images import *
X_train,Y_train=prepare_dataset_for_images(505,28,14,"C:/Users/pramesh/PycharmProjects/BODMASS/dataset/train/")#762
X_test,Y_test=prepare_dataset_for_images(22,28,14,"C:/Users/pramesh/PycharmProjects/BODMASS/dataset/test/")
X_train=X_train/255
X_test=X_test/255

model(X_train,Y_train,X_test,Y_test)
