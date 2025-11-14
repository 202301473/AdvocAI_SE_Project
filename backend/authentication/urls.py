from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    signup_view, 
    login_view, 
    google_auth_view, 
    profile_view, 
    logout_view,
    verify_otp_view,
    resend_otp_view
)

urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('google/', google_auth_view, name='google_auth'),
    path('verify-otp/', verify_otp_view, name='verify_otp'),
    path('resend-otp/', resend_otp_view, name='resend_otp'),
    path('profile/', profile_view, name='profile'),
    path('logout/', logout_view, name='logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
