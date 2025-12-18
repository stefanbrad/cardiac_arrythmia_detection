// Check if user is already logged in
if (api.isAuthenticated()) {
    window.location.href = 'dashboard.html';
}

// Login form handler
document.getElementById('loginForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const errorMessage = document.getElementById('errorMessage');
    const loginBtn = e.target.querySelector('.login-btn');
    
    // Hide previous error messages
    errorMessage.style.display = 'none';
    
    // Disable button during login
    loginBtn.disabled = true;
    loginBtn.textContent = 'Logging in...';
    
    try {
        // Call API login
        await api.login(username, password);
        
        // Redirect to dashboard on success
        window.location.href = 'dashboard.html';
        
    } catch (error) {
        // Show error message
        errorMessage.textContent = 'Invalid username or password. Please try again.';
        errorMessage.style.display = 'block';
        
        // Re-enable button
        loginBtn.disabled = false;
        loginBtn.textContent = 'Login';
        
        // Clear password field
        document.getElementById('password').value = '';
    }
});

// Enter key support for password field
document.getElementById('password').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        document.getElementById('loginForm').dispatchEvent(new Event('submit'));
    }
});