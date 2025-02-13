# MNIST Handwritten Digit Classification

<p>
    This project aims to detect handwritten digit images using Neural Networks. The neural network is build using keras, imported from Tensorflow library. The model gives <b>97.67%</b> accuracy on the test data. 
<br/>
<br/>
    Also, this is deployed using a streamlit web app, where a user can upload an image of a handwritten digit and the model predicts the digit.
<br/>
<br/>
    <b>Deployed link: <a href="https://garimauttam-handwritten-digit-classification-using-n-app-hlfe9z.streamlit.app/">Click here to visit the application</a></b>
</p>

<hr/>
<h3>Technologies used in this project: </h3>
<ul>
    <li>numpy</li>
    <li>matplotlib</li>
    <li>seaborn</li>
    <li>opencv-python</li>
    <li>Pillow</li>
    <li>tensorflow</li>
    <li>streamlit</li>
</ul>

<hr/>
<h3>Steps to run this project: </h3>

<p><b>Step-1: </b> Create an empty directory and run the following commands in the terminal: </p>

```bash
git clone https://github.com/GarimaUttam/Handwritten_Digit_Classification_using_Neural_Network.git

cd mnist-handwritten-digit-classification
```

<p><b>Step-2: </b>Create a virtual environment and install the dependencies.
<br/>
    <b>Note: </b> This example is given provided you are using a Windows OS. If you have any other, please follow the standard guide from documentation.
</p>

```bash
python -m venv env

.\env\Scripts\activate

pip install -r requiremnts.txt
```

<p><b>Step-3: </b>Run the code.ipynb file selecting the kernel from the virtual environment created in the previous step. A <b>model.h5</b> file is created in the root folder after successful execution.</p>

<p><b>Step-4: </b> Run the streamlit app using the following command</p>

```bash
streamlit run .\app.py
```

<p><b>Step-5: </b> Select a handwritten digit image and upload. You will see the predicted output. </p>
