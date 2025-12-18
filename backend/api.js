// API Base URL - Change this to your backend URL
const API_BASE_URL = 'http://localhost:5000/api';

// API Helper Functions
const api = {
    // Login endpoint
    login: async (username, password) => {
        try {
            console.log('Attempting login with:', username); // Add this line
            const response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password })
            });

            console.log('Response status:', response.status); // Add this line
            if (!response.ok) {
                const errorData = await response.json();
                console.log('Error data:', errorData); // Add this line
                throw new Error('Login failed');
            }

            const data = await response.json();
            
            // Store token in sessionStorage
            if (data.token) {
                sessionStorage.setItem('authToken', data.token);
                sessionStorage.setItem('username', username);
            }
            
            return data;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    },

    // Logout endpoint
    logout: async () => {
        try {
            const token = sessionStorage.getItem('authToken');
            
            await fetch(`${API_BASE_URL}/auth/logout`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                }
            });

            // Clear session storage
            sessionStorage.clear();
        } catch (error) {
            console.error('Logout error:', error);
            sessionStorage.clear();
        }
    },

    // Upload and analyze ECG file
    analyzeECG: async (file, patientData) => {
        try {
            const token = sessionStorage.getItem('authToken');
            const formData = new FormData();
            
            formData.append('ecg_file', file);
            formData.append('patient_id', patientData.id || '');
            formData.append('age', patientData.age || '');
            formData.append('gender', patientData.gender || '');

            const response = await fetch(`${API_BASE_URL}/analysis/analyze`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Analysis error:', error);
            throw error;
        }
    },

    // Get analysis results
    getResults: async (analysisId) => {
        try {
            const token = sessionStorage.getItem('authToken');
            
            const response = await fetch(`${API_BASE_URL}/analysis/results/${analysisId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to get results');
            }

            return await response.json();
        } catch (error) {
            console.error('Get results error:', error);
            throw error;
        }
    },

    // Download report
    downloadReport: async (analysisId) => {
        try {
            const token = sessionStorage.getItem('authToken');
            
            const response = await fetch(`${API_BASE_URL}/analysis/report/${analysisId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to download report');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ecg_report_${analysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Download error:', error);
            throw error;
        }
    },

    // Save analysis
    saveAnalysis: async (analysisId, patientData) => {
        try {
            const token = sessionStorage.getItem('authToken');
            
            const response = await fetch(`${API_BASE_URL}/analysis/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    analysis_id: analysisId,
                    patient_data: patientData
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save analysis');
            }

            return await response.json();
        } catch (error) {
            console.error('Save analysis error:', error);
            throw error;
        }
    },

    // Check authentication
    isAuthenticated: () => {
        return sessionStorage.getItem('authToken') !== null;
    },

    // Get current username
    getCurrentUser: () => {
        return sessionStorage.getItem('username');
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
}