const API_BASE_URL = 'http://localhost:5000/api';

// API Functions
const api = {
    // login endpoint
    login: async (username, password) => {
        try {
            console.log('Attempting login with:', username); 
            const response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password })
            });

            console.log('Response status:', response.status); 
            if (!response.ok) {
                const errorData = await response.json();
                console.log('Error data:', errorData); 
                throw new Error('Login failed');
            }

            const data = await response.json();
            
            // store token in sessionStorage
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

    // logout endpoint
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

            // clear session storage
            sessionStorage.clear();
        } catch (error) {
            console.error('Logout error:', error);
            sessionStorage.clear();
        }
    },

    // upload and analyze ECG file
analyzeECG: async (recordId, patientData) => {
        try {
            const token = sessionStorage.getItem('authToken');
            
            // building json payload
            const bodyPayload = {
                record_id: recordId,
                patient_id: patientData.id || '',
                age: patientData.age || '',
                gender: patientData.gender || ''
            };

            const response = await fetch(`${API_BASE_URL}/analysis/analyze`, {
                method: 'POST',
                headers: {
                    // !!! ACEASTA LINIE E FFF IMPORTANTA !!!
                    'Content-Type': 'application/json', 
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(bodyPayload) 
            });

            // handling errors
            if (!response.ok) {
                const errorData = await response.json(); 
                console.log('Server Error:', errorData);
                throw new Error(errorData.error || 'Analysis failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Analysis error:', error);
            throw error;
        }
    },

    // get analysis results
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

    // download report (mai am de lucrat aici)
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

    // save analysis (mai am de lucrat aici again)
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

    // check authentication
    isAuthenticated: () => {
        return sessionStorage.getItem('authToken') !== null;
    },

    // get current username
    getCurrentUser: () => {
        return sessionStorage.getItem('username');
    }
};

// export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
}