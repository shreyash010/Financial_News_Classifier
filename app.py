from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load and preprocess the model
model = pickle.load(open('train_model.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])
    flag = prediction[0]
    if(flag == 0):
        return render_template('result.html', prediction = "negative")
    elif(flag == 1):
        return render_template('result.html', prediction = "neutral")
    return render_template('result.html', prediction= "positive");


if __name__ == '__main__':
    app.run(debug=True)
