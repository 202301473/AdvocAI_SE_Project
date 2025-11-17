from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from django.http import FileResponse
import markdown
from xhtml2pdf import pisa
from io import BytesIO
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import cloudinary.uploader
from documents.mongo_client import get_conversation_by_id


