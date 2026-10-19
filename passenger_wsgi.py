"""Phusion Passenger WSGI entry point for cPanel/o2switch deployments.

Configure the Python application startup file as passenger_wsgi.py and
entry point as application. Set ARTIST_DNA_API_TOKEN in the hosting
environment before exposing the analysis endpoints.
"""
from a2wsgi import ASGIMiddleware

from app import create_app

asgi_app = create_app()
application = ASGIMiddleware(asgi_app)
