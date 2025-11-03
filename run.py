from app import app

if __name__ == '__main__':
    # Run on localhost
    app.run(debug=True)
    # Or run on specific host (uncomment line below)
    # app.run(debug=True, host="192.168.1.87")
