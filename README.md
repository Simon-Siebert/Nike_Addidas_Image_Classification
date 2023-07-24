<h1>Nike and Addidas Image Classification</h1>

<h2>Description</h2>
This is a deep learning project that demonstrates image classification using a custom CNN architecture. The model is trained to classify images of Nike and Adidas shoes. The goal of this project is to develop a deep learning model capable of accurately classifying images of Nike and Adidas shoes. The model is built using TensorFlow Keras and consists of three main parts: Entry Flow, Middle Flow, and Exit Flow.
<br />

<h2>Dataset</h2>
<p>This dataset is the Nike,Adidas Shoes for Image Classification Dataset found on Kaggle. The dataset had to many seperate files to upload to GitHub, so here is the link to the dataset. https://www.kaggle.com/datasets/ifeanyinneji/nike-adidas-shoes-for-image-classification-dataset/code?select=labelnames.csv</p>
<h2>Languages and Methods Used</h2>

- <b>Python</b> 
- <b>Tensorflow</b>
- <b>Keras</b>
- <b>Pandas</b>
- <b>Numpy</b>
- <b>Pillow</b>
- <b>Matplotlib</b>
- <b>Seaborn</b>

<h2>IDE Used </h2>

- <b>VS Code</b>

<h2>Model Architecture</h2>
The model architecture is a custom CNN based on the concepts of Convolutional Neural Networks. It comprises three main parts:
<br />
<li>Entry Flow: The initial part of the model that performs feature extraction from the input image.</li>
<li>Middle Flow: Repeated layers to capture more complex features.</li>
<li>Exit Flow: Final layers to map features to the output class (Nike or Adidas).</li>

<h2>Training</h2>
The model is trained using an Adam optimizer with a learning rate of 1e-3 and binary cross-entropy as the loss function. Data augmentation is performed on the training set to improve the model's generalization.

<h2>Evaluation</h2>
The trained model is evaluated on a separate testing set to measure its accuracy in classifying images of Nike and Adidas shoes.

<h2>Results</h2>
The model's performance, including accuracy and confusion matrix, is reported after evaluation on the testing set.
