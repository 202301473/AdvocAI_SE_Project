import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser

from documents.mongo_client import get_all_conversations, get_conversation_by_id, save_conversation, update_conversation, delete_conversation, get_document_version_content, delete_document_version, update_share_permissions, update_user_share_permissions
from channels.layers import get_channel_layer # Import get_channel_layer
from asgiref.sync import async_to_sync # Import async_to_sync
from ai_generator.utils import get_gemini_response # Import the AI generation function


from .comment_mongo_client import get_comments_for_document, add_comment, serialize_comment


