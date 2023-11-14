import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import sys
import re
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

plt.style.use('ggplot')
def plot_example(X, y):
    """Plot the first 100 images in a 10x10 grid."""
    plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)

    for i in range(10):  # For 10 rows
        for j in range(10):  # For 10 columns
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    plt.savefig('output_plot.png')
    #plt.show()  # Display the entire grid

def visualize_data(X, y,n):
    """Plot the first 100 images in a 10x10 grid."""
    plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)
    n=int(n**(0.5))
    for i in range(n):  # For 10 rows
        for j in range(n):  # For 10 columns
            index = i * n + j
            plt.subplot(n, n, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=16)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    plt.savefig('output_plot.png')
    #plt.show()  # Display the entire grid

# Loading Data
#@st.cache_data
def load_model(model_name):
    mnist = fetch_openml(model_name, as_frame=False, cache=False)
    return mnist

mnist = load_model("mnist_784")

# Preprocessing Data
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# to avoid big weights
X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
assert(X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])


selected2 = option_menu(None, ["About", "Dataset", "Model", "Train","Predict"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2== "Predict":
    # Build Neural Network with PyTorch
    st.markdown("""
                <h1 style='text-align: center;'>Make Prediction on Your Data</h1>
                <p align="justify">After learning from train data, You friend computer can predict what number is written in the picture. You can upload an image and can test your model on your own data image.</p>
    """,True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(mnist.target))

    # A Neural network in PyTorch's framework.
    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
            ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)

            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X
    
    test_image=st.file_uploader("upload an image")
    if test_image is not None:
        test_image=Image.open(test_image)
        test_image=test_image.resize((128,128))
        st.image(test_image)
        image_np=np.array(test_image)
        threshold = 200
        white_pixels = np.all(image_np >= threshold, axis=-1)
        white_pixel_count = np.sum(white_pixels)
        total_pixels = np.prod(image_np.shape[:2])
        white_percentage = (white_pixel_count / total_pixels) * 100
        if white_percentage > 60:
            html_str = f"""
            <style>
            h3 {{
                color: blue;
            }}
            </style>
            <h3>The image has a white background.</h3>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            invert_image = np.where(image_np >= 200, 30,170)
            test_image =Image.fromarray(invert_image.astype(np.uint8))
        else:
            html_str = f"""
            <style>
            h3 {{
                color: blue;
            }}
            </style>
            <h3>The image has a colorful background (not white).</h3>
            """
            st.markdown(html_str, unsafe_allow_html=True)
        test_image=test_image.convert("L")
        st.image(test_image)
        test_image=test_image.resize((28,28))
        test_image=np.array(test_image)
        test_image=test_image.flatten()
        test_image=test_image.astype('float32')      
        normalized_test_image = (test_image / 255.0)
        from skorch import NeuralNetClassifier
        torch.manual_seed(0)
        net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=10,
            lr=0.1,
            device=device,
        )
        net.fit(X_train, y_train)
        X_test[0]=normalized_test_image
        y_pred=net.predict(X_test)
        html_str = f"""
        <style>
        h1 {{
            color: red;
        }}
        </style>
        <h1>Model Prediction: <span style="color:blue;">{y_pred[0]} </span></h1>
        <h2> Is the model prediction correct? </h2>
        """
        st.markdown(html_str, unsafe_allow_html=True)
        st.selectbox("select",["yes","No"])
elif selected2=="Dataset" :
    st.markdown("""
                <h1 style='text-align: center;'>Create Your Own Dataset</h1>
    """,True)
    st.markdown(""" We want to teach our computer friend to recognize handwritten numbers from 0 to 9. We'll show it many examples of each digit and tell it, "This is how I write 0, 1, 2, and so on." Once our computer friend learns, we can give it a new handwritten number, and it will happily tell us which digit it thinks it is! :sunglasses: """,True)
    st.markdown("""
                <h2>MNIST</h2>
                <p align="justify" >Imagine MNIST is like a big collection of pictures of handwritten numbers. It's like a special album that helps computers learn how to recognize numbers written by people. So, when a computer looks at these pictures from MNIST, it learns to tell which number is in each picture, just like you learn to recognize different letters and numbers in school!</p>
                <p align="justify" >Here, Your friend have such an album which contains pictures of handwritten digits. But he is not opened it yet. You can see some pictures by click on VISUALIZE DATA</p>
    """,True)

    if st.button("Show data size"):
        st.write("Album (MNIST) has a collection of ",mnist.data.shape[0], "pictures.")
    val=st.select_slider("No. of Sample",options=[4,9,16,25,36,49,64,81,100])
    vd = st.button("VISUALIZE DATA")
    if vd:
        visualize_data(X_train, y_train,val)
        st.image("output_plot.png")
    st.markdown("""
                <h2>Train-Test Split</h2>
                <p align="justify" >First of all, You select samples (already selected 60000) from that data album and then split them in train data and test data. For training you friend computer, You have to show all the pictures from training data and tell it that It is 0 or 1 or 2 and so on and your freind learns curves of digit, shape and something else. Now, You can test your friend to show pictures from test data.</p>
    """,True)
    split_ratio=st.slider("Split Ratio",0.1,0.9,0.7)
    vd1 = st.button("Generate DATA")
    if vd1:
        X_Sample,Y_sample,x_sample,y_sample = train_test_split(X, y,test_size=10000, random_state=42)
        X_train,X_test,y_train,y_test = train_test_split(X_Sample, x_sample,test_size=1-split_ratio, random_state=42)
        d={ 
            "Train Data":X_train.shape[0],
            "Test Data":X_test.shape[0],
            "Total Sample Data":X_Sample.shape[0],
        }
        st.table(d)
elif selected2=="Train" :
    st.markdown("""
                <h1 style='text-align: center;'>Train Your Model</h1>
                <p align="justify" >For teaching your friend computer to recognize your handwritten numbers, like the ones you draw on paper, first, you show your friend lots of examples of the numbers you have written, telling it, "This is the number 5, and this is how I draw it. Now, your friend practices looking at all these examples and learns to understand the special shapes and lines that make each number unique. Once your friend feels confident, you give it a new drawing, and it happiliy shouts out the number it thinks you wrote!" </p>
                <h2>Train The Network</h2>
                <p align="justify" >Your friend computer has a smart brain. It has two parts: one for understanding the features of the numbers (like curves and lines), and another for making predictions. It teach itself using a special set of pictures ( train data) and their labels to get better at recognizing the numbers. It does this training process for a maximum of 20 times(epochs), adjusting and improving each time. It uses a learing rate of 0.1, which helps it decide to change itself during each learning step to make a better prediction.</p>
    """,True)
    # Build Neural Network with PyTorch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(mnist.target))

    # A Neural network in PyTorch's framework.
    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
            ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)

            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X
    
    # skorch allows to use PyTorch's networks in the SciKit-Learn setting:
    btn=st.button("Train Model")
    if btn:
        from skorch import NeuralNetClassifier
        torch.manual_seed(0)
        net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=20,
            lr=0.1,
            device=device,
        )
        original_stdout = sys.stdout
        with open('filename.txt','w') as f:
            sys.stdout=f
            net.fit(X_train, y_train)
            sys.stdout=original_stdout
        filename = 'filename.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
        data = [re.split(r'\s+', re.sub(r'\x1b\[[0-9;]*m', '', line.strip())) for line in lines[2:]]
        columns = ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']
        df = pd.DataFrame(data, columns=columns)
        st.write(df)
        plt.plot(df['epoch'], df['valid_acc'], marker='o')
        plt.title('Epoch vs. Valid Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Valid Accuracy')
        plt.grid(True)
        plt.savefig('m1.png')
        st.image('m1.png')
elif selected2=="Model":
    st.markdown("""
                <h1 style='text-align: center;'>Build Your Own Model</h1>
    """,True)
    st.markdown(""" 
                Building your own model is like training your magical friend to recognize hand-written numbers. You show it lots of examples, and it learns to guess the right number when you draw something new. It's like having a drawing detective friend!:sunglasses: 
    """,True)
    st.markdown("""
                <h2>Design a Neural Network</h2>
                <p align="justify" >Your friend first looks at your drawing, which is a picture made up of tiny dots, just like the pixels on your computer screen. Your friend has such a brain where are hidden helpers (neurons) that work together. Each helper is like a detective looking for specific things in your drawing, like curves, lines or corners. Before becoming a super smart friend, it needs some training. You show it many drawings of numbers. and each time, it adjusts its helpers to get better at recognizing patterns. Now, when you draw a new number, it's helpers quickly analyze the drawing and vote on what they think it is. The most votes decide what your friend predicts! If your friend makes a correct prediction, it cheers, "Yay, I got it right!" If not, It learns from the mistake so it can do better next time.</p>
    """,True)
    st.image("data//model.png")
else:
    st.markdown("""
            <h1 style='text-align: center;'>Introduction to Image Classification</h1>
            <p align="justify" >Can you know that a computer can tell us which digit we have written by looking at it? But how ? Imagine you have a bunch of pictures, like photos of cats, 
            and dogs. Now, image classification is like teaching a
            computer to recognize which picture is of a cat and which
            one is of a dog.
            </p>  
            <h2>How Computers See</h2>
            <p align="justify" >Imagine the computer is like a super smart friend with special eyes that can look at pictures. But this friend does not know what things are, like cats or dogs. So, you get to teach them! <br> you show your friend lots of pictures and say, "Look, this is a cat and this is a dog." The computer learns what makes a cat look like a cat and what make a dog like a dog based on ears, tails, and fur patterns. Now, when you show the computer a new picture, it tries to decide if it is a cat or dog based on what it learned.</p>
    """,True)
    st.image("data//intro.png")

# # # Build Neural Network with PyTorch
# import torch
# from torch import nn
# import torch.nn.functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mnist_dim = X.shape[1]
# hidden_dim = int(mnist_dim/8)
# output_dim = len(np.unique(mnist.target))

# print(mnist_dim, hidden_dim, output_dim)

# # A Neural network in PyTorch's framework.
# class ClassifierModule(nn.Module):
#     def __init__(
#             self,
#             input_dim=mnist_dim,
#             hidden_dim=hidden_dim,
#             output_dim=output_dim,
#             dropout=0.5,
#     ):
#         super(ClassifierModule, self).__init__()
#         self.dropout = nn.Dropout(dropout)

#         self.hidden = nn.Linear(input_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, output_dim)

#     def forward(self, X, **kwargs):
#         X = F.relu(self.hidden(X))
#         X = self.dropout(X)
#         X = F.softmax(self.output(X), dim=-1)
#         return X
    
# #skorch allows to use PyTorch's networks in the SciKit-Learn setting:

# from skorch import NeuralNetClassifier

# torch.manual_seed(0)

# net = NeuralNetClassifier(
#     ClassifierModule,
#     max_epochs=20,
#     lr=0.1,
#     device=device,
# )
# net.fit(X_train, y_train)

# # prediction
# from sklearn.metrics import accuracy_score
# y_pred = net.predict(X_test)
# accuracy_score(y_test, y_pred)

# error_mask = y_pred != y_test
# plot_example(X_test[error_mask], y_pred[error_mask])

# # convolutional network

# XCnn = X.reshape(-1, 1, 28, 28)

# print(XCnn.shape)
# XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)
# print(XCnn_train.shape, y_train.shape)

# class Cnn(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(Cnn, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d(p=dropout)
#         self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
#         self.fc2 = nn.Linear(100, 10)
#         self.fc1_drop = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x = torch.relu(F.max_pool2d(self.conv1(x), 2))
#         x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

#         # flatten over channel, height and width = 1600
#         x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

#         x = torch.relu(self.fc1_drop(self.fc1(x)))
#         x = torch.softmax(self.fc2(x), dim=-1)
#         return x
    
# torch.manual_seed(0)

# cnn = NeuralNetClassifier(
#     Cnn,
#     max_epochs=10,
#     lr=0.002,
#     optimizer=torch.optim.Adam,
#     device=device,
# )

# cnn.fit(XCnn_train, y_train)

# y_pred_cnn = cnn.predict(XCnn_test)
# print(accuracy_score(y_test, y_pred_cnn))

# print(accuracy_score(y_test[error_mask], y_pred_cnn[error_mask]))

# plot_example(X_test[error_mask], y_pred_cnn[error_mask])