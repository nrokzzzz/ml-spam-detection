from flask import Flask,url_for,redirect,render_template,request,session
from joblib import load
app = Flask(__name__)
app.secret_key = 'your_secret_key'
vectorizer = load('vectorizer.joblib')
modelNaive = load('modelNaive.joblib')

@app.route('/')
def home():
    res = session.pop('res', '')
    return render_template('index.html',res=res)

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        text=request.form['data']
        transformed = vectorizer.transform([text])
        prediction = modelNaive.predict(transformed)[0]
        if prediction== 1:
            session['res']="⚠️ Warning: Be careful! Spam messages may contain harmful content or phishing links. Stay alert and protect your personal info."
        elif prediction==0:
            session['res']="It does not contain any signs of spam or harmful content. You can review and respond as needed. Still, always be cautious with links and attachments."
    return redirect(url_for('home'))
if __name__ == '__main__':
    app.run(debug=True)

