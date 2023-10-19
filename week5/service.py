from flask import Flask


app = Flask('model-service')

@app.route('/pred', methods=['GET'])
def get_pred():
    return 'Hello World'
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

