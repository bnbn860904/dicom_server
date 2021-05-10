from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "hi!"


@app.route('/user/<username>')
def username(username):
    answer = 'i am ' + username
    
    return answer


@app.route('/age/<int:age>')
def userage(age):
    return 'i am ' + str(age) + 'years old'


if __name__ == "__main__":
    app.run()