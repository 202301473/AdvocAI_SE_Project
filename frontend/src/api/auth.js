import axiosInstance from './axios';

export const authAPI = {
  // Manual signup
  signup: async (userData) => {
    const response = await axiosInstance.post('/auth/signup/', userData);
    return response.data;
  },

  // Manual login
  login: async (credentials) => {
    const response = await axiosInstance.post('/auth/login/', credentials);
    return response.data;
  },

  // Google OAuth
  googleAuth: async (token) => {
    const response = await axiosInstance.post('/auth/google/', { token });
    return response.data;
  },

  // Get user profile
  getProfile: async () => {
    const response = await axiosInstance.get('/auth/profile/');
    return response.data;
  },

  // Logout
  logout: async (refreshToken) => {
    const response = await axiosInstance.post('/auth/logout/', {
      refresh: refreshToken,
    });
    return response.data;
  },

  // Verify OTP
  verifyOTP: async (data) => {
    const response = await axiosInstance.post('/auth/verify-otp/', data);
    return response.data;
  },

  // Resend OTP
  resendOTP: async (data) => {
    const response = await axiosInstance.post('/auth/resend-otp/', data);
    return response.data;
  },
};
