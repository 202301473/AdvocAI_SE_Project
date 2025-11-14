from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('download-pdf/', views.download_pdf, name='download_pdf'),
    path('preview/', views.preview_document, name='preview_document'),
]
