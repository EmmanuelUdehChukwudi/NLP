from flask import Flask, request, jsonify,render_template
from utils import get_response ,predict_class

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    response = get_response(predict_class(user_message))
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host= "0.0.0.0",port=5000, debug=True)
    
    