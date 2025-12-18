from flask import Flask
from flask_cors import CORS
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from routes.auth import bp as auth_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')

@app.route('/')
def index():
    return jsonify({
        'message': 'Cardiac Arrhythmia Detection API',
        'version': '1.0.0',
        'status': 'online'
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)