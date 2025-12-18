from flask import Blueprint, request, jsonify
import jwt
from datetime import datetime, timedelta
from functools import wraps

bp = Blueprint('auth', __name__)

# Secret key for JWT (should match app.py)
SECRET_KEY = 'your-secret-key-here'

# Dummy user database (replace with actual database)
users_db = {
    'admin': 'admin123',
    'doctor': 'doctor123',
    'user': 'password123',
    'newuser': 'newpassword',  # Add new users like this
    'john': 'john123'          # Add as many as you need
}

# Store active tokens (in production, use Redis or database)
active_tokens = set()

def token_required(f):
    """Decorator to protect routes with JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        if token not in active_tokens:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user = data['username']
        except jwt.ExpiredSignatureError:
            active_tokens.discard(token)
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

@bp.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Missing username or password'}), 400
        
        # Verify credentials (replace with database check)
        if username in users_db and users_db[username] == password:
            # Generate JWT token
            token = jwt.encode({
                'username': username,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, SECRET_KEY, algorithm='HS256')
            
            # Store token
            active_tokens.add(token)
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'username': username
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout endpoint"""
    try:
        token = request.headers['Authorization'].split(' ')[1]
        active_tokens.discard(token)
        
        return jsonify({'message': 'Logout successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/verify', methods=['GET'])
@token_required
def verify(current_user):
    """Verify token endpoint"""
    return jsonify({
        'valid': True,
        'username': current_user
    }), 200